#!/usr/bin/env python3
"""
Vidar - Speed Detection App

Detects object speed based on relative frame-to-frame size changes
using webcam input and YOLO object detection.

Usage:
    python main.py                    # Use default webcam
    python main.py --iphone           # Use iPhone Continuity Camera
    python main.py --camera 1         # Use specific camera index
    python main.py --list-cameras     # List available cameras
    python main.py video.mp4          # Use video file

Controls:
    - Click on an object to select it as primary (for detailed tracking)
    - Press 'c' to clear selection
    - Press 'q' to quit
"""

import sys
import argparse
import cv2

from src.camera import CameraCapture, VideoFileCapture, list_available_cameras
from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.speed_calculator import SpeedCalculator
from src.ui import render_frame


class VidarApp:
    """Main application class for Vidar speed detection."""

    def __init__(self, video_path: str = None, camera_index: int = None, use_iphone: bool = False):
        self.video_path = video_path
        self.using_video = video_path is not None

        # Load YOLO first (this was the original order)
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.speed_calc = SpeedCalculator()

        # Now initialize camera - iPhone may need YOLO load time to connect
        if self.using_video:
            self.camera = VideoFileCapture(video_path)
        else:
            if camera_index is not None:
                self.camera = CameraCapture(camera_index=camera_index)
            else:
                self.camera = CameraCapture(use_iphone=use_iphone)

        self.window_name = "Vidar - Speed Detection"
        self.running = False
        self.show_depth = False  # Depth visualization toggle

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_id = self.tracker.select_primary((x, y))
            if selected_id is not None:
                print(f"Selected object #{selected_id}")

    def run(self):
        """Main application loop."""
        print("Starting Vidar Speed Detection...")
        print("=" * 40)

        # Initialize camera
        if not self.camera.start():
            print("Failed to start camera. Exiting.")
            sys.exit(1)

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\nControls:")
        print("  - Click on object to track")
        print("  - Press 'd' to toggle depth view")
        print("  - Press 'c' to clear selection")
        print("  - Press 'q' to quit")
        print("=" * 40)

        self.running = True
        failed_frames = 0
        max_failed_frames = 30  # Allow some failed frames before giving up

        try:
            while self.running:
                # Read frame
                ret, frame, timestamp = self.camera.read()
                if not ret or frame is None:
                    failed_frames += 1
                    if failed_frames >= max_failed_frames:
                        print(f"Failed to read {max_failed_frames} consecutive frames. Exiting.")
                        break
                    continue

                failed_frames = 0  # Reset counter on successful read

                # Detect objects
                detections = self.detector.detect(frame)

                # Update tracker
                tracked_objects = self.tracker.update(detections, timestamp)

                # Calculate speed metrics for all tracked objects
                speed_metrics = {}
                for obj_id, tracked_obj in tracked_objects.items():
                    metrics = self.speed_calc.calculate(tracked_obj)
                    if metrics is not None:
                        speed_metrics[obj_id] = metrics

                # Render frame with overlays
                display_frame = render_frame(
                    frame,
                    tracked_objects,
                    speed_metrics,
                    self.camera.get_fps(),
                    show_depth=self.show_depth
                )

                # Show frame
                cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Quit requested")
                    self.running = False
                elif key == ord('c'):
                    print("Cleared selection")
                    self.tracker.clear()
                    self.speed_calc.reset()
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                    print(f"Depth view: {'ON' if self.show_depth else 'OFF'}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Done.")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Vidar - Speed Detection using object tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default webcam
  python main.py --iphone           # Use iPhone Continuity Camera
  python main.py --camera 1         # Use camera at index 1
  python main.py --list-cameras     # List all available cameras
  python main.py video.mp4          # Process video file
        """
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Video file path (optional, uses camera if not provided)"
    )
    parser.add_argument(
        "--iphone", "-i",
        action="store_true",
        help="Use iPhone Continuity Camera (macOS only)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        help="Camera index to use (default: 0)"
    )
    parser.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="List available cameras and exit"
    )

    args = parser.parse_args()

    # List cameras mode
    if args.list_cameras:
        print("\nDiscovering cameras...")
        cameras = list_available_cameras()
        if not cameras:
            print("No cameras found.")
        else:
            print(f"\nFound {len(cameras)} camera(s):\n")
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['name']}")
                print(f"      Resolution: {cam['resolution']}")
                print(f"      Backend: {cam['backend']}")
                print()
        print("Use --camera <index> to select a specific camera")
        print("Use --iphone to auto-detect iPhone Continuity Camera")
        sys.exit(0)

    # Run app
    app = VidarApp(
        video_path=args.video,
        camera_index=args.camera,
        use_iphone=args.iphone
    )
    app.run()


if __name__ == "__main__":
    main()
