"""Feature transforms from the upstream demo; now guarded against invalid input."""
import math


def _points(points, expected=None):
    points = [list(point) for point in points]
    if expected is not None and len(points) != expected:
        raise ValueError(f'Expected {expected} points, got {len(points)}')
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('Points must contain two finite coordinates')
    return points


def normalize_landmarks(points):
    """21 (x, y) pairs -> 42 wrist-relative values scaled by max magnitude."""
    points = _points(points, expected=21)
    base_x, base_y = points[0]
    flat = [value for x, y in points for value in (x - base_x, y - base_y)]
    scale = max(map(abs, flat))
    return [v / scale for v in flat] if scale else [0.0] * 42


def normalize_history(points, width, height):
    """Make a fingertip trajectory relative to its first point and frame size."""
    if width <= 0 or height <= 0:
        raise ValueError('Frame dimensions must be positive')
    points = _points(points)
    if not points:
        return []
    base_x, base_y = points[0]
    return [v for x, y in points for v in ((x - base_x) / width, (y - base_y) / height)]
