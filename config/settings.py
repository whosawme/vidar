"""Configuration constants for Vidar speed detection app."""

# Camera settings
CAMERA_INDEX = 0  # Default camera index
CAMERA_WIDTH = 1920  # iPhone Continuity Camera supports up to 1920x1080
CAMERA_HEIGHT = 1080
TARGET_FPS = 30  # iPhone cameras can do 30fps reliably

# iPhone Continuity Camera notes:
# - On macOS 13+, iPhone appears as a camera when nearby and on same Apple ID
# - Uses AVFoundation backend automatically on macOS
# - Supports high resolution (1080p) and good frame rates
# - Use --iphone flag or --list-cameras to find your iPhone

# Object detection settings
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed
DETECTION_CONFIDENCE = 0.5
DETECTION_IOU_THRESHOLD = 0.45

# Tracking settings
TRACKING_IOU_THRESHOLD = 0.3  # Min IoU to consider same object
MAX_FRAMES_TO_SKIP = 10  # Max frames an object can be missing before losing track
HISTORY_LENGTH = 30  # Number of frames to keep in history

# Speed calculation settings
SMOOTHING_ALPHA = 0.3  # Exponential moving average factor (0-1, lower = smoother)
MIN_AREA_CHANGE = 100  # Minimum area change to register (noise filter)
MAX_SPEED_NORMALIZATION = 50000  # px²/s for 100% relative speed

# Calibration defaults (for real speed estimation)
KNOWN_OBJECT_WIDTH_M = 0.5  # Default known object width in meters
FOCAL_LENGTH_PX = 800  # Approximate focal length in pixels (needs calibration)

# UI settings
BOX_THICKNESS = 2
PRIMARY_COLOR = (0, 255, 0)  # Green BGR
SECONDARY_COLOR = (255, 165, 0)  # Orange BGR
TEXT_COLOR = (255, 255, 255)  # White
FONT_SCALE = 0.6
SPEED_GAUGE_RADIUS = 80
SPEED_GAUGE_POSITION = (100, 100)  # Top-left position

# Color gradient for speed (BGR format)
SPEED_COLORS = {
    "low": (0, 255, 0),      # Green - slow
    "medium": (0, 255, 255),  # Yellow - medium
    "high": (0, 0, 255),      # Red - fast
}
