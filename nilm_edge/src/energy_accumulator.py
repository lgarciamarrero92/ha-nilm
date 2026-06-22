"""Persistent, Home Assistant-style energy integration for NILM power entities."""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict


DEFAULT_MAX_INTEGRATION_GAP_S = 300.0


@dataclass(frozen=True)
class EnergyUpdate:
    total_kwh: float
    integration_gap_s: float
    skipped_stale_gap: bool


class EnergyAccumulator:
    """Accumulate power readings with trapezoidal integration.

    The state file is intentionally independent from the add-on configuration so
    installations upgrading from older versions start safely with empty state.
    Gaps longer than ``max_integration_gap_s`` are not integrated: the previous
    reading is replaced with a new baseline instead of assuming it held during
    an unknown period.
    """

    def __init__(self, path: str, max_integration_gap_s: float = DEFAULT_MAX_INTEGRATION_GAP_S):
        self.path = path
        self.max_integration_gap_s = float(max_integration_gap_s)
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            entries = payload.get("appliances", {}) if isinstance(payload, dict) else {}
            if not isinstance(entries, dict):
                return
            for key, value in entries.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                total_kwh = float(value.get("total_kwh", 0.0))
                previous_power_w = float(value["previous_power_w"])
                previous_timestamp = float(value["previous_timestamp"])
                if (
                    math.isfinite(total_kwh)
                    and total_kwh >= 0.0
                    and math.isfinite(previous_power_w)
                    and math.isfinite(previous_timestamp)
                ):
                    self._entries[key] = {
                        "total_kwh": total_kwh,
                        "previous_power_w": max(0.0, previous_power_w),
                        "previous_timestamp": previous_timestamp,
                    }
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load NILM energy state: {exc}")

    def _save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self.path}.tmp"
        payload = {"version": 1, "appliances": self._entries}
        try:
            with open(temporary_path, "w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, separators=(",", ":"), allow_nan=False)
            os.replace(temporary_path, self.path)
        except OSError as exc:
            print(f"Warning: could not save NILM energy state: {exc}")
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    def update(self, appliance_key: str, power_w: float, timestamp_s: float) -> EnergyUpdate:
        """Record one published power reading and return the cumulative energy."""
        power_w = max(0.0, float(power_w))
        timestamp_s = float(timestamp_s)
        if not math.isfinite(power_w) or not math.isfinite(timestamp_s):
            raise ValueError("Energy integration requires finite power and timestamp values")

        entry = self._entries.get(appliance_key)
        if entry is None:
            entry = {
                "total_kwh": 0.0,
                "previous_power_w": power_w,
                "previous_timestamp": timestamp_s,
            }
            self._entries[appliance_key] = entry
            self._save()
            return EnergyUpdate(total_kwh=0.0, integration_gap_s=0.0, skipped_stale_gap=False)

        elapsed_s = timestamp_s - float(entry["previous_timestamp"])
        if elapsed_s <= 0.0:
            return EnergyUpdate(
                total_kwh=float(entry["total_kwh"]),
                integration_gap_s=elapsed_s,
                skipped_stale_gap=False,
            )

        skipped_stale_gap = elapsed_s > self.max_integration_gap_s
        if not skipped_stale_gap:
            average_power_w = (float(entry["previous_power_w"]) + power_w) / 2.0
            entry["total_kwh"] = float(entry["total_kwh"]) + (average_power_w * elapsed_s / 3_600_000.0)

        entry["previous_power_w"] = power_w
        entry["previous_timestamp"] = timestamp_s
        self._save()
        return EnergyUpdate(
            total_kwh=float(entry["total_kwh"]),
            integration_gap_s=elapsed_s,
            skipped_stale_gap=skipped_stale_gap,
        )
