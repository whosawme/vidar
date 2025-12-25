"""UI overlay and display module for Vidar speed detection app."""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from src.tracker import TrackedObject
from src.speed_calculator import SpeedMetrics, SpeedDirection
from config.settings import (
    BOX_THICKNESS,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    TEXT_COLOR,
    FONT_SCALE,
    SPEED_GAUGE_RADIUS,
    SPEED_GAUGE_POSITION,
    SPEED_COLORS
)


def get_speed_color(relative_speed: float) -> Tuple[int, int, int]:
    """Get color based on speed (green -> yellow -> red gradient).

    Args:
        relative_speed: Speed value from 0.0 to 1.0

    Returns:
        BGR color tuple
    """
    if relative_speed < 0.33:
        return SPEED_COLORS["low"]
    elif relative_speed < 0.66:
        # Interpolate between green and yellow
        t = (relative_speed - 0.33) / 0.33
        low = SPEED_COLORS["low"]
        mid = SPEED_COLORS["medium"]
        return (
            int(low[0] + t * (mid[0] - low[0])),
            int(low[1] + t * (mid[1] - low[1])),
            int(low[2] + t * (mid[2] - low[2]))
        )
    else:
        # Interpolate between yellow and red
        t = (relative_speed - 0.66) / 0.34
        mid = SPEED_COLORS["medium"]
        high = SPEED_COLORS["high"]
        return (
            int(mid[0] + t * (high[0] - mid[0])),
            int(mid[1] + t * (high[1] - mid[1])),
            int(mid[2] + t * (high[2] - mid[2]))
        )


def draw_bounding_box(
    frame: np.ndarray,
    tracked_obj: TrackedObject,
    metrics: Optional[SpeedMetrics] = None
) -> np.ndarray:
    """Draw bounding box around tracked object.

    Args:
        frame: Image to draw on
        tracked_obj: TrackedObject to draw
        metrics: Optional speed metrics for coloring

    Returns:
        Frame with drawing
    """
    bbox = tracked_obj.detection.bbox
    x1, y1, x2, y2 = bbox

    # Choose color based on primary status and speed
    if tracked_obj.is_primary:
        if metrics is not None:
            color = get_speed_color(metrics.relative_speed)
        else:
            color = PRIMARY_COLOR
        thickness = BOX_THICKNESS + 1
    else:
        color = SECONDARY_COLOR
        thickness = BOX_THICKNESS

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    label = f"ID:{tracked_obj.id} {tracked_obj.detection.class_name}"
    if metrics is not None:
        label += f" {metrics.direction_symbol}"

    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
    )

    cv2.rectangle(
        frame,
        (x1, y1 - label_h - 10),
        (x1 + label_w + 5, y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (0, 0, 0),  # Black text on colored background
        1,
        cv2.LINE_AA
    )

    # Draw mini speed badge for non-primary objects
    if not tracked_obj.is_primary and metrics is not None:
        badge_text = f"{metrics.relative_percent:.0f}%"
        (badge_w, badge_h), _ = cv2.getTextSize(
            badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        badge_x = x2 - badge_w - 5
        badge_y = y1 + badge_h + 5

        cv2.rectangle(
            frame,
            (badge_x - 2, badge_y - badge_h - 2),
            (badge_x + badge_w + 2, badge_y + 2),
            get_speed_color(metrics.relative_speed),
            -1
        )
        cv2.putText(
            frame,
            badge_text,
            (badge_x, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_speed_gauge(
    frame: np.ndarray,
    metrics: SpeedMetrics,
    position: Tuple[int, int] = SPEED_GAUGE_POSITION,
    radius: int = SPEED_GAUGE_RADIUS
) -> np.ndarray:
    """Draw circular speed gauge for primary object.

    Args:
        frame: Image to draw on
        metrics: Speed metrics to display
        position: Center position of gauge
        radius: Radius of gauge

    Returns:
        Frame with gauge
    """
    cx, cy = position
    color = get_speed_color(metrics.relative_speed)

    # Draw background circle
    cv2.circle(frame, (cx, cy), radius, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), radius, (100, 100, 100), 2)

    # Draw speed arc (0-270 degrees based on speed)
    angle = int(270 * metrics.relative_speed)
    if angle > 0:
        cv2.ellipse(
            frame,
            (cx, cy),
            (radius - 5, radius - 5),
            -90,
            0,
            angle,
            color,
            8
        )

    # Draw speed text in center
    speed_text = f"{metrics.relative_percent:.0f}%"
    (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame,
        speed_text,
        (cx - tw // 2, cy + th // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    # Draw direction arrow
    arrow_y = cy + radius // 2
    cv2.putText(
        frame,
        metrics.direction_symbol,
        (cx - 8, arrow_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
    )

    return frame


def draw_metrics_panel(
    frame: np.ndarray,
    metrics: SpeedMetrics,
    tracked_obj: TrackedObject
) -> np.ndarray:
    """Draw detailed metrics panel.

    Args:
        frame: Image to draw on
        metrics: Speed metrics to display
        tracked_obj: Tracked object info

    Returns:
        Frame with panel
    """
    h, w = frame.shape[:2]

    # Panel dimensions
    panel_w = 280
    panel_h = 140
    panel_x = w - panel_w - 10
    panel_y = 10

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (30, 30, 30),
        -1
    )
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # Draw border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        get_speed_color(metrics.relative_speed),
        2
    )

    # Draw title
    title = f"Object #{tracked_obj.id}: {tracked_obj.detection.class_name}"
    cv2.putText(
        frame,
        title,
        (panel_x + 10, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )

    # Draw metrics
    y_offset = panel_y + 50
    line_height = 25

    # Relative speed
    cv2.putText(
        frame,
        f"Relative: {metrics.relative_percent:.1f}%",
        (panel_x + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        get_speed_color(metrics.relative_speed),
        1,
        cv2.LINE_AA
    )
    y_offset += line_height

    # Size change rate
    cv2.putText(
        frame,
        f"Rate: {metrics.size_change_rate:+.0f} px²/s",
        (panel_x + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )
    y_offset += line_height

    # Estimated speed
    if metrics.estimated_speed_kmh is not None:
        speed_text = f"Est. Speed: {metrics.estimated_speed_kmh:.1f} km/h"
    else:
        speed_text = "Est. Speed: -- km/h"
    cv2.putText(
        frame,
        speed_text,
        (panel_x + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )
    y_offset += line_height

    # Direction
    dir_text = f"Direction: {metrics.direction.value} {metrics.direction_symbol}"
    cv2.putText(
        frame,
        dir_text,
        (panel_x + 10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        get_speed_color(metrics.relative_speed),
        1,
        cv2.LINE_AA
    )

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter.

    Args:
        frame: Image to draw on
        fps: Frames per second

    Returns:
        Frame with FPS
    """
    h, w = frame.shape[:2]
    fps_text = f"FPS: {fps:.1f}"

    cv2.putText(
        frame,
        fps_text,
        (w - 100, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    return frame


def draw_instructions(frame: np.ndarray) -> np.ndarray:
    """Draw usage instructions.

    Args:
        frame: Image to draw on

    Returns:
        Frame with instructions
    """
    h, w = frame.shape[:2]

    instructions = [
        "Click on object to track",
        "Press 'q' to quit",
        "Press 'c' to clear selection"
    ]

    y = h - 80
    for instruction in instructions:
        cv2.putText(
            frame,
            instruction,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )
        y += 20

    return frame


def render_frame(
    frame: np.ndarray,
    tracked_objects: Dict[int, TrackedObject],
    speed_metrics: Dict[int, SpeedMetrics],
    fps: float
) -> np.ndarray:
    """Render complete frame with all overlays.

    Args:
        frame: Original camera frame
        tracked_objects: Dictionary of tracked objects
        speed_metrics: Dictionary of speed metrics per object
        fps: Current FPS

    Returns:
        Rendered frame with all overlays
    """
    # Draw all bounding boxes
    primary_obj = None
    primary_metrics = None

    for obj_id, tracked_obj in tracked_objects.items():
        metrics = speed_metrics.get(obj_id)
        frame = draw_bounding_box(frame, tracked_obj, metrics)

        if tracked_obj.is_primary:
            primary_obj = tracked_obj
            primary_metrics = metrics

    # Draw primary object gauge and metrics
    if primary_obj is not None and primary_metrics is not None:
        frame = draw_speed_gauge(frame, primary_metrics)
        frame = draw_metrics_panel(frame, primary_metrics, primary_obj)

    # Draw FPS and instructions
    frame = draw_fps(frame, fps)
    frame = draw_instructions(frame)

    return frame
