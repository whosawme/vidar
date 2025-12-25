"""Object detection module using YOLO for Vidar speed detection app."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

from config.settings import (
    YOLO_MODEL,
    DETECTION_CONFIDENCE,
    DETECTION_IOU_THRESHOLD
)


@dataclass
class Detection:
    """Represents a single detected object."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class ObjectDetector:
    """YOLO-based object detector."""

    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        confidence: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU_THRESHOLD
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Model loaded. Classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Detection objects
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, "unknown")

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(detection)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return self.class_names.get(class_id, "unknown")
