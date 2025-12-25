"""Camera capture module for Vidar speed detection app."""

import cv2
import time
from typing import Optional, Tuple, List, Dict
from config.settings import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS


def list_available_cameras(max_cameras: int = 10) -> List[Dict]:
    """Discover available cameras.

    Returns:
        List of dicts with camera info: {index, name, backend, resolution}
    """
    cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend = cap.getBackendName()

            name = f"Camera {i}"
            if w >= 1920 and h >= 1080:
                name = f"Camera {i} (HD)"

            cameras.append({
                "index": i,
                "name": name,
                "backend": backend,
                "resolution": f"{w}x{h}"
            })
            cap.release()

    return cameras


class CameraCapture:
    """Handles webcam capture and frame timing."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        target_fps: int = TARGET_FPS,
        use_iphone: bool = False
    ):
        # If use_iphone requested, try to find a non-zero camera index
        if use_iphone:
            cameras = list_available_cameras()
            if len(cameras) > 1:
                # Use the second camera (likely iPhone)
                camera_index = cameras[1]["index"]
                print(f"Using camera at index {camera_index}")
            else:
                print("Only one camera found, using default camera 0")
                camera_index = 0

        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time: float = 0
        self.actual_fps: float = 0
        self._fps_samples: list = []

    def start(self) -> bool:
        """Initialize and start the camera capture.

        Returns:
            True if camera started successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Get actual resolution (may differ from requested)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera started: {actual_width}x{actual_height}")

        # Warmup: read a few frames to let camera initialize
        print("Warming up camera...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        self.last_frame_time = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[any], float]:
        """Read a frame from the camera.

        Returns:
            Tuple of (success, frame, timestamp)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0

        ret, frame = self.cap.read()
        current_time = time.time()

        if ret:
            # Calculate FPS
            dt = current_time - self.last_frame_time
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                self.actual_fps = sum(self._fps_samples) / len(self._fps_samples)

            self.last_frame_time = current_time

        return ret, frame, current_time

    def get_fps(self) -> float:
        """Get the current actual FPS."""
        return self.actual_fps

    def stop(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class VideoFileCapture:
    """Handles video file capture for testing."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_time: float = 0
        self.actual_fps: float = 0
        self.video_fps: float = 30
        self._fps_samples: list = []

    def start(self) -> bool:
        """Initialize and start the video capture."""
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False

        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded: {self.video_path}")
        print(f"Resolution: {width}x{height}, FPS: {self.video_fps:.1f}, Frames: {frame_count}")

        self.last_frame_time = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[any], float]:
        """Read a frame from the video."""
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0

        ret, frame = self.cap.read()
        current_time = time.time()

        # Loop video when it ends
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if ret:
            dt = current_time - self.last_frame_time
            if dt > 0:
                self._fps_samples.append(1.0 / dt)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                self.actual_fps = sum(self._fps_samples) / len(self._fps_samples)

            self.last_frame_time = current_time

        return ret, frame, current_time

    def get_fps(self) -> float:
        """Get the current actual FPS."""
        return self.actual_fps

    def stop(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Video stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
