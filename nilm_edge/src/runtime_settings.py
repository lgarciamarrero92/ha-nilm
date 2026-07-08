DEFAULT_SENSOR_MAX_GAP_S = 60.0
MIN_SENSOR_MAX_GAP_S = 1.0
MAX_SENSOR_MAX_GAP_S = 3600.0


def clamp_sensor_max_gap_s(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SENSOR_MAX_GAP_S
    return max(MIN_SENSOR_MAX_GAP_S, min(MAX_SENSOR_MAX_GAP_S, parsed))
