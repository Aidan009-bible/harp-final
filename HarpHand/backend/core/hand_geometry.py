from __future__ import annotations


def adaptive_touch_distance_px(
    width: int,
    height: int,
    *,
    ratio: float = 0.0185,
    minimum_px: float = 6.0,
    maximum_px: float = 32.0,
) -> float:
    """Scale touch distance by the shorter frame edge with practical bounds."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive")
    if ratio <= 0 or minimum_px <= 0 or maximum_px < minimum_px:
        raise ValueError("Touch-distance configuration is invalid")
    return max(minimum_px, min(maximum_px, min(width, height) * ratio))


def normalized_touch_confidence(distance_px: float, threshold_px: float) -> float:
    if threshold_px <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - max(0.0, distance_px) / threshold_px))
