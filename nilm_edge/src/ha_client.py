# src/ha_client.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from aiohttp import ClientSession, ClientError


@dataclass(frozen=True)
class HistoryQuery:
    entity_id: str
    start_dt: datetime
    end_dt: datetime
    minimal_response: bool = True
    max_span_days: int = 7


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_z(dt: datetime, *, timespec: str = "seconds") -> str:
    """
    Format datetime as ISO8601 ending in 'Z' (UTC).
    """
    dt = _ensure_utc(dt)
    return dt.isoformat(timespec=timespec).replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """
    Robustly parse ISO 8601 timestamps that may include 'Z'.
    """
    # Handles strings like 2025-12-15T11:33:53Z
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def validate_history_range(start_dt: datetime, end_dt: datetime, max_span_days: int = 7) -> Tuple[datetime, datetime]:
    """
    Enforce:
      - start < end
      - <= max_span_days of actual duration
    """
    start_dt = _ensure_utc(start_dt)
    end_dt = _ensure_utc(end_dt)

    if start_dt >= end_dt:
        raise ValueError("Start date must be before end date.")

    max_span = max_span_days * 24 * 60 * 60
    span_seconds = (end_dt - start_dt).total_seconds()
    if span_seconds > max_span + 1:
        raise ValueError(f"Date range cannot exceed {max_span_days} days.")

    return start_dt, end_dt


async def fetch_history_raw(
    ha_rest_api_url: str,
    token: str,
    query: HistoryQuery,
) -> list:
    """
    Fetch raw history JSON payload from Home Assistant for a given HistoryQuery.
    Raises:
      - ValueError for invalid date ranges
      - RuntimeError for HA fetch errors
    """
    start_dt, end_dt = validate_history_range(query.start_dt, query.end_dt, query.max_span_days)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    start_iso = _iso_z(start_dt, timespec="seconds")
    end_iso = _iso_z(end_dt, timespec="milliseconds")

    url = (
        f"{ha_rest_api_url}/history/period/{start_iso}"
        f"?end_time={end_iso}"
        f"&filter_entity_id={query.entity_id}"
        f"&minimal_response={'true' if query.minimal_response else 'false'}"
    )

    async with ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except ClientError as e:
            raise RuntimeError(f"Could not fetch history from Home Assistant: {e}") from e


async def fetch_history_points(
    ha_rest_api_url: str,
    token: str,
    query: HistoryQuery,
) -> List[Tuple[float, float]]:
    """
    Fetch history and return a sorted list of (epoch_seconds, value).

    Robustness:
      - skips malformed entries
      - returns [] if empty or unexpected format
      - raises ValueError for invalid ranges
      - raises RuntimeError for HA client errors
    """
    raw_history = await fetch_history_raw(ha_rest_api_url, token, query)

    points: List[Tuple[float, float]] = []

    if not raw_history or not isinstance(raw_history, list) or len(raw_history) == 0:
        return points

    # We requested a single entity, HA typically returns [ [states...] ]
    sensor_history = raw_history[0] if raw_history and raw_history[0] else []

    for state_obj in sensor_history:
        try:
            # A power sensor may refresh its state without changing the numeric
            # value. Prefer last_updated so those fresh samples are not mistaken
            # for a long recorder gap.
            timestamp_str = state_obj.get("last_updated") or state_obj.get("last_changed")
            value_str = state_obj.get("state")

            if not timestamp_str or value_str is None:
                continue

            if isinstance(value_str, str):
                lowered = value_str.strip().lower()
                if lowered == "on":
                    value = 1.0
                elif lowered == "off":
                    value = 0.0
                else:
                    value = float(value_str)
            else:
                value = float(value_str)
            ts = _parse_iso_utc(timestamp_str).timestamp()

            points.append((ts, value))
        except (ValueError, TypeError):
            # Skip malformed history entries (same robustness you had)
            continue
        except Exception:
            continue

    points.sort(key=lambda p: p[0])
    return points


async def call_ha_service(
    ha_rest_api_url: str,
    token: str,
    domain: str,
    service: str,
    service_data: dict,
) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    url = f"{ha_rest_api_url}/services/{domain}/{service}"

    async with ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=service_data) as response:
                response.raise_for_status()
                if response.content_type == "application/json":
                    return await response.json()
                return {"ok": True}
        except ClientError as e:
            raise RuntimeError(f"Could not call Home Assistant service {domain}.{service}: {e}") from e


def points_to_xy_json(
    points: List[Tuple[float, float]],
    *,
    expand_held_values: bool = False,
    end_dt: Optional[datetime] = None,
) -> List[dict]:
    """
    Convert (epoch_seconds, value) -> [{"x": "...Z", "y": ...}, ...]

    When ``expand_held_values`` is enabled, duplicate the current value right
    before each subsequent timestamp so UIs that render plain line charts see
    the Home Assistant history as held power levels instead of diagonal ramps.
    """
    out: List[dict] = []
    if not points:
        return out

    sorted_points = sorted(points, key=lambda p: p[0])
    epsilon_s = 0.001

    for index, (ts, val) in enumerate(sorted_points):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        out.append({"x": _iso_z(dt, timespec="milliseconds"), "y": float(val)})

        if not expand_held_values or index >= len(sorted_points) - 1:
            continue

        next_ts = float(sorted_points[index + 1][0])
        hold_ts = next_ts - epsilon_s
        if hold_ts > ts:
            hold_dt = datetime.fromtimestamp(hold_ts, tz=timezone.utc)
            out.append({"x": _iso_z(hold_dt, timespec="milliseconds"), "y": float(val)})

    if expand_held_values and end_dt is not None:
        end_ts = _ensure_utc(end_dt).timestamp()
        last_ts, last_val = sorted_points[-1]
        if end_ts > last_ts:
            out.append({"x": _iso_z(_ensure_utc(end_dt), timespec="milliseconds"), "y": float(last_val)})

    return out
