"""Object tracking module for Vidar speed detection app."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from src.detector import Detection
from config.settings import (
    TRACKING_IOU_THRESHOLD,
    MAX_FRAMES_TO_SKIP,
    HISTORY_LENGTH
)


@dataclass
class SizeSnapshot:
    """Records object size at a point in time."""
    timestamp: float
    area: int
    bbox: Tuple[int, int, int, int]


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    id: int
    detection: Detection
    first_seen: float
    last_seen: float
    frames_missing: int = 0
    is_primary: bool = False
    history: List[SizeSnapshot] = field(default_factory=list)

    def update(self, detection: Detection, timestamp: float):
        """Update tracked object with new detection."""
        self.detection = detection
        self.last_seen = timestamp
        self.frames_missing = 0

        # Add to history
        snapshot = SizeSnapshot(
            timestamp=timestamp,
            area=detection.area,
            bbox=detection.bbox
        )
        self.history.append(snapshot)

        # Trim history to limit
        if len(self.history) > HISTORY_LENGTH:
            self.history = self.history[-HISTORY_LENGTH:]

    @property
    def age(self) -> float:
        """Time since first detection in seconds."""
        return self.last_seen - self.first_seen

    @property
    def current_area(self) -> int:
        """Current bounding box area."""
        return self.detection.area

    @property
    def center(self) -> Tuple[int, int]:
        """Current center position."""
        return self.detection.center


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes.

    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class ObjectTracker:
    """Tracks objects across frames using IoU matching."""

    def __init__(
        self,
        iou_threshold: float = TRACKING_IOU_THRESHOLD,
        max_frames_to_skip: int = MAX_FRAMES_TO_SKIP
    ):
        self.iou_threshold = iou_threshold
        self.max_frames_to_skip = max_frames_to_skip

        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.primary_id: Optional[int] = None

    def update(self, detections: List[Detection], timestamp: float) -> Dict[int, TrackedObject]:
        """Update tracked objects with new detections.

        Args:
            detections: List of detections from current frame
            timestamp: Current frame timestamp

        Returns:
            Dictionary of tracked objects
        """
        # Mark all objects as potentially missing
        for obj in self.tracked_objects.values():
            obj.frames_missing += 1

        # Match detections to existing tracks
        matched_detection_indices = set()

        for det_idx, detection in enumerate(detections):
            best_match_id = None
            best_iou = self.iou_threshold

            # Find best matching tracked object
            for obj_id, tracked_obj in self.tracked_objects.items():
                iou = calculate_iou(detection.bbox, tracked_obj.detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id

            if best_match_id is not None:
                # Update existing track
                self.tracked_objects[best_match_id].update(detection, timestamp)
                matched_detection_indices.add(det_idx)
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1

                new_track = TrackedObject(
                    id=new_id,
                    detection=detection,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    history=[SizeSnapshot(
                        timestamp=timestamp,
                        area=detection.area,
                        bbox=detection.bbox
                    )]
                )
                self.tracked_objects[new_id] = new_track
                matched_detection_indices.add(det_idx)

        # Remove tracks that have been missing too long
        ids_to_remove = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.frames_missing > self.max_frames_to_skip:
                ids_to_remove.append(obj_id)

        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
            if self.primary_id == obj_id:
                self.primary_id = None

        return self.tracked_objects

    def select_primary(self, click_pos: Tuple[int, int]) -> Optional[int]:
        """Select primary object based on click position.

        Args:
            click_pos: (x, y) position of click

        Returns:
            ID of selected object, or None if no object at position
        """
        x, y = click_pos

        for obj_id, tracked_obj in self.tracked_objects.items():
            bbox = tracked_obj.detection.bbox
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                # Clear previous primary
                if self.primary_id is not None and self.primary_id in self.tracked_objects:
                    self.tracked_objects[self.primary_id].is_primary = False

                # Set new primary
                self.primary_id = obj_id
                tracked_obj.is_primary = True
                return obj_id

        return None

    def get_primary(self) -> Optional[TrackedObject]:
        """Get the primary tracked object."""
        if self.primary_id is not None and self.primary_id in self.tracked_objects:
            return self.tracked_objects[self.primary_id]
        return None

    def get_all_objects(self) -> List[TrackedObject]:
        """Get all tracked objects."""
        return list(self.tracked_objects.values())

    def clear(self):
        """Clear all tracked objects."""
        self.tracked_objects.clear()
        self.primary_id = None
