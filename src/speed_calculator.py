"""Speed calculation module for Vidar speed detection app."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import math

from src.tracker import TrackedObject, SizeSnapshot
from config.settings import (
    SMOOTHING_ALPHA,
    MIN_AREA_CHANGE,
    MAX_SPEED_NORMALIZATION,
    KNOWN_OBJECT_WIDTH_M,
    FOCAL_LENGTH_PX
)


class SpeedDirection(Enum):
    """Direction of object movement relative to camera."""
    APPROACHING = "approaching"
    RECEDING = "receding"
    STABLE = "stable"


@dataclass
class SpeedMetrics:
    """Speed measurement results."""
    relative_speed: float  # 0.0 to 1.0 (normalized)
    size_change_rate: float  # pixels²/second (raw)
    estimated_speed_ms: Optional[float]  # meters/second (calibrated)
    direction: SpeedDirection

    @property
    def relative_percent(self) -> float:
        """Relative speed as percentage (0-100)."""
        return self.relative_speed * 100

    @property
    def estimated_speed_kmh(self) -> Optional[float]:
        """Estimated speed in km/h."""
        if self.estimated_speed_ms is not None:
            return self.estimated_speed_ms * 3.6
        return None

    @property
    def direction_symbol(self) -> str:
        """Arrow symbol indicating direction."""
        if self.direction == SpeedDirection.APPROACHING:
            return "↑"
        elif self.direction == SpeedDirection.RECEDING:
            return "↓"
        return "•"


class SpeedCalculator:
    """Calculates speed metrics from tracked object size changes."""

    def __init__(
        self,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        min_area_change: float = MIN_AREA_CHANGE,
        max_speed_norm: float = MAX_SPEED_NORMALIZATION,
        known_object_width: float = KNOWN_OBJECT_WIDTH_M,
        focal_length: float = FOCAL_LENGTH_PX
    ):
        self.smoothing_alpha = smoothing_alpha
        self.min_area_change = min_area_change
        self.max_speed_norm = max_speed_norm
        self.known_object_width = known_object_width
        self.focal_length = focal_length

        # Smoothed values per object
        self._smoothed_rates: dict = {}
        self._max_observed_rate: float = max_speed_norm

    def calculate(self, tracked_obj: TrackedObject) -> Optional[SpeedMetrics]:
        """Calculate speed metrics for a tracked object.

        Args:
            tracked_obj: TrackedObject with history

        Returns:
            SpeedMetrics or None if insufficient history
        """
        history = tracked_obj.history

        if len(history) < 2:
            return None

        # Calculate raw size change rate using recent history
        raw_rate = self._calculate_raw_rate(history)

        # Apply exponential smoothing
        obj_id = tracked_obj.id
        if obj_id not in self._smoothed_rates:
            self._smoothed_rates[obj_id] = raw_rate
        else:
            self._smoothed_rates[obj_id] = (
                self.smoothing_alpha * raw_rate +
                (1 - self.smoothing_alpha) * self._smoothed_rates[obj_id]
            )

        smoothed_rate = self._smoothed_rates[obj_id]

        # Update max observed rate for normalization
        abs_rate = abs(smoothed_rate)
        if abs_rate > self._max_observed_rate:
            self._max_observed_rate = abs_rate

        # Determine direction
        if abs(smoothed_rate) < self.min_area_change:
            direction = SpeedDirection.STABLE
        elif smoothed_rate > 0:
            direction = SpeedDirection.APPROACHING
        else:
            direction = SpeedDirection.RECEDING

        # Calculate relative speed (normalized 0-1)
        relative_speed = min(1.0, abs_rate / self._max_observed_rate)

        # Calculate estimated real speed
        estimated_speed = self._estimate_real_speed(history, smoothed_rate)

        return SpeedMetrics(
            relative_speed=relative_speed,
            size_change_rate=smoothed_rate,
            estimated_speed_ms=estimated_speed,
            direction=direction
        )

    def _calculate_raw_rate(self, history: List[SizeSnapshot]) -> float:
        """Calculate raw size change rate from history.

        Uses weighted average of recent changes for stability.
        """
        if len(history) < 2:
            return 0.0

        # Use last few snapshots for averaging
        num_samples = min(5, len(history) - 1)
        total_rate = 0.0
        total_weight = 0.0

        for i in range(1, num_samples + 1):
            curr = history[-i]
            prev = history[-(i + 1)]

            dt = curr.timestamp - prev.timestamp
            if dt > 0:
                area_change = curr.area - prev.area
                rate = area_change / dt

                # Weight recent measurements more heavily
                weight = 1.0 / i
                total_rate += rate * weight
                total_weight += weight

        if total_weight > 0:
            return total_rate / total_weight
        return 0.0

    def _estimate_real_speed(
        self,
        history: List[SizeSnapshot],
        size_change_rate: float
    ) -> Optional[float]:
        """Estimate real-world speed using pinhole camera model.

        The pinhole camera model relates object size to distance:
        distance = (known_width * focal_length) / apparent_width

        Speed is then the rate of change of distance.
        """
        if len(history) < 2:
            return None

        try:
            curr = history[-1]
            prev = history[-2]

            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                return None

            # Get apparent widths (from bbox)
            curr_width = curr.bbox[2] - curr.bbox[0]
            prev_width = prev.bbox[2] - prev.bbox[0]

            if curr_width <= 0 or prev_width <= 0:
                return None

            # Calculate distances using pinhole model
            curr_distance = (self.known_object_width * self.focal_length) / curr_width
            prev_distance = (self.known_object_width * self.focal_length) / prev_width

            # Speed is rate of distance change
            speed = abs(curr_distance - prev_distance) / dt

            return speed

        except (ZeroDivisionError, IndexError):
            return None

    def calibrate(self, known_width_m: float, focal_length_px: Optional[float] = None):
        """Update calibration parameters.

        Args:
            known_width_m: Known object width in meters
            focal_length_px: Camera focal length in pixels (optional)
        """
        self.known_object_width = known_width_m
        if focal_length_px is not None:
            self.focal_length = focal_length_px

    def reset(self):
        """Reset smoothing state."""
        self._smoothed_rates.clear()
        self._max_observed_rate = self.max_speed_norm

    def remove_object(self, obj_id: int):
        """Remove smoothing state for an object."""
        self._smoothed_rates.pop(obj_id, None)
