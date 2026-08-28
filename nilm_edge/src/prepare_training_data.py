from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from runtime_settings import DEFAULT_SENSOR_MAX_GAP_S

try:
    # Preferred for HA app (lightweight)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for dev machines with full TF installed
    from tensorflow.lite import Interpreter


# -----------------------
# Settings
# -----------------------

@dataclass(frozen=True)
class ModelSettings:
    version: int
    saved_at: str
    sequence_length: int
    frequency_s: float
    pred_idx: int
    query_mean: float
    query_std: float
    ref_mean: float
    ref_std: float
    power_mean: float
    power_std: float


def load_model_settings(settings_path: str) -> ModelSettings:
    with open(settings_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    seq_len = int(raw["sequence_length"])
    pred_idx = int(raw.get("pred_idx", seq_len - 1))
    if pred_idx < 0 or pred_idx >= seq_len:
        raise ValueError(f"pred_idx out of range: {pred_idx} for sequence_length={seq_len}")

    return ModelSettings(
        version=int(raw.get("version", 1)),
        saved_at=str(raw.get("saved_at", "")),
        sequence_length=seq_len,
        frequency_s=float(raw["frequency"]),
        pred_idx=pred_idx,
        query_mean=float(raw["query_mean"]),
        query_std=float(raw["query_std"]),
        ref_mean=float(raw["ref_mean"]),
        ref_std=float(raw["ref_std"]),
        power_mean=float(raw["power_mean"]),
        power_std=float(raw["power_std"]),
    )


# -----------------------
# Robust parsing & intervals
# -----------------------

def _ms_to_s(ms: float) -> float:
    return float(ms) / 1000.0


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def normalize_on_intervals(selectedWindows: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    UI selectedWindows are ON segments. Convert to merged, sorted intervals in epoch seconds.
    Each element expects: {"start": ms_epoch, "end": ms_epoch, ...}
    """
    intervals: List[Tuple[float, float]] = []
    for w in selectedWindows:
        try:
            a = _ms_to_s(float(w["start"]))
            b = _ms_to_s(float(w["end"]))
            s, e = (a, b) if a < b else (b, a)
            if e > s and np.isfinite(s) and np.isfinite(e):
                intervals.append((s, e))
        except Exception:
            continue
    return _merge_intervals(intervals)


def parse_mains_points(fullSensorHistoryData: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    UI fullSensorHistoryData: [{x: ms_epoch, y: watts}, ...]
    Robust parsing: drop invalid, sort, de-dupe by timestamp (keep last).
    Returns: list of (t_s, watts) sorted by time.
    """
    tmp: List[Tuple[float, float]] = []
    for d in fullSensorHistoryData:
        try:
            t_ms = d.get("x", None)
            y = d.get("y", None)
            if t_ms is None or y is None:
                continue
            t_s = _ms_to_s(float(t_ms))
            val = float(y)
            if not np.isfinite(t_s) or not np.isfinite(val):
                continue
            tmp.append((t_s, val))
        except Exception:
            continue

    if not tmp:
        return []

    tmp.sort(key=lambda p: p[0])

    out: List[Tuple[float, float]] = []
    last_t: Optional[float] = None
    for t, v in tmp:
        if last_t is None or t != last_t:
            out.append((t, v))
            last_t = t
        else:
            out[-1] = (t, v)  # keep last
    return out


def parse_power_points(sensorHistoryData: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    Generic parser for appliance power history traces.
    Uses the same UI payload shape as mains: [{x: ms_epoch, y: watts}, ...]
    """
    return parse_mains_points(sensorHistoryData)


# -----------------------
# Robust resampling
# -----------------------

def build_uniform_grid(t_start: float, t_end: float, dt: float, align: str = "start") -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_end <= t_start:
        return np.zeros((0,), dtype=np.float64)

    n = int(np.floor((t_end - t_start) / dt)) + 1
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)

    if align == "end":
        grid0 = t_start + np.arange(n, dtype=np.float64) * dt
        last = grid0[-1]
        remainder = (t_end - last)
        shift = remainder if 0.0 <= remainder < dt else 0.0
        return grid0 + shift

    return t_start + np.arange(n, dtype=np.float64) * dt


def zoh_resample_to_grid(
    pts: List[Tuple[float, float]],
    grid_t: np.ndarray,
    *,
    max_hold_s: Optional[float] = None,
    fill_value: float = 0.0,
    return_valid_mask: bool = False,
) -> Any:
    """
    ZOH resample: each grid point takes last observed value.
    If max_hold_s is set and last observed is older than that, output fill_value.
    """
    if grid_t.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        if return_valid_mask:
            return empty, np.zeros((0,), dtype=bool)
        return empty
    if not pts:
        out = np.full((len(grid_t),), float(fill_value), dtype=np.float32)
        valid_mask = np.zeros((len(grid_t),), dtype=bool)
        if return_valid_mask:
            return out, valid_mask
        return out

    t = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)

    idx = np.searchsorted(t, grid_t, side="right") - 1

    out = np.full((len(grid_t),), float(fill_value), dtype=np.float32)
    valid = idx >= 0
    idxc = np.clip(idx, 0, len(t) - 1)

    out[valid] = y[idxc[valid]].astype(np.float32)
    valid_mask = valid.copy()

    if max_hold_s is not None:
        age = grid_t - t[idxc]
        stale = age > float(max_hold_s)
        out[stale] = float(fill_value)
        valid_mask[stale] = False

    if return_valid_mask:
        return out, valid_mask
    return out


def summarize_mains_validity(
    pts: List[Tuple[float, float]],
    valid_mask: np.ndarray,
    *,
    sample_period_s: float,
    max_hold_s: Optional[float],
) -> Dict[str, float]:
    """Return concise diagnostics for rejected training windows."""
    mask = np.asarray(valid_mask, dtype=bool)
    longest_run_samples = 0
    current_run_samples = 0
    for is_valid in mask:
        if is_valid:
            current_run_samples += 1
            longest_run_samples = max(longest_run_samples, current_run_samples)
        else:
            current_run_samples = 0

    timestamps = np.asarray([point[0] for point in pts], dtype=np.float64)
    raw_gaps = np.diff(timestamps) if timestamps.size > 1 else np.zeros((0,), dtype=np.float64)
    stale_gaps = raw_gaps[raw_gaps > float(max_hold_s)] if max_hold_s is not None else np.zeros((0,), dtype=np.float64)
    return {
        "invalid_samples": float(np.count_nonzero(~mask)),
        "longest_valid_span_s": float(max(0, longest_run_samples - 1) * sample_period_s),
        "max_raw_gap_s": float(np.max(raw_gaps)) if raw_gaps.size else 0.0,
        "stale_gap_count": float(stale_gaps.size),
        "max_stale_gap_s": float(np.max(stale_gaps)) if stale_gaps.size else 0.0,
    }




# -----------------------
# Labeling at pred_idx
# -----------------------

def _is_on_at_time(t: float, intervals: List[Tuple[float, float]], start_idx: int) -> Tuple[int, int]:
    i = start_idx
    while i < len(intervals) and intervals[i][1] <= t:
        i += 1
    if i < len(intervals) and intervals[i][0] <= t < intervals[i][1]:
        return 1, i
    return 0, i


def compute_on_mask(
    ap_seq: np.ndarray,
    *,
    sample_period_s: float,
    threshold_watts: float,
    window_hours: float,
    min_on_s: float,
    min_off_s: float,
) -> np.ndarray:
    x = np.asarray(ap_seq, dtype=np.float32).ravel()
    x[np.isnan(x)] = 0.0
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    win = max(1, int(round(float(window_hours) * 3600.0 / float(sample_period_s))))
    if win % 2 == 0:
        win += 1
    pad = win // 2

    def moving_min_centered(a: np.ndarray, window: int, pad_: int) -> np.ndarray:
        ap = np.pad(a, (pad_, pad_), mode="edge")
        dq = deque()
        out = np.empty(ap.size - window + 1, dtype=a.dtype)
        for i, val in enumerate(ap):
            while dq and ap[dq[-1]] >= val:
                dq.pop()
            dq.append(i)
            if dq[0] <= i - window:
                dq.popleft()
            if i >= window - 1:
                out[i - (window - 1)] = ap[dq[0]]
        return out

    baseline = moving_min_centered(x, win, pad)
    detrended = x - baseline
    detrended[detrended < 0.0] = 0.0
    mask = (detrended > float(threshold_watts)).astype(np.uint8)

    def to_samples_ge1(seconds: float) -> int:
        return max(1, int(round(float(seconds) / float(sample_period_s))))

    min_on_samples = to_samples_ge1(min_on_s)
    bridge_off_samples = to_samples_ge1(min_off_s)

    def rle_runs(b: np.ndarray):
        i = 0
        while i < n:
            v = b[i]
            j = i + 1
            while j < n and b[j] == v:
                j += 1
            yield int(v), i, j
            i = j

    if bridge_off_samples > 0:
        for v, s, e in list(rle_runs(mask)):
            if v == 0 and (e - s) < bridge_off_samples:
                left_on = (s > 0) and (mask[s - 1] == 1)
                right_on = (e < n) and (mask[e] == 1)
                if left_on and right_on:
                    mask[s:e] = 1

    if min_on_samples > 1:
        for v, s, e in list(rle_runs(mask)):
            if v == 1 and (e - s) < min_on_samples:
                mask[s:e] = 0

    return mask.astype(np.uint8)


# -----------------------
# TFLite extractor wrapper
# -----------------------

class QueryExtractor:
    def __init__(self, model_path: str, num_threads: int = 2, eps: float = 1e-6):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Extractor model not found: {model_path}")

        self.interp = Interpreter(model_path=model_path, num_threads=int(num_threads))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()
        self.out = self.interp.get_output_details()

        if len(self.inp) != 1:
            raise RuntimeError(f"Extractor expected 1 input, got {len(self.inp)}")
        if len(self.out) < 1:
            raise RuntimeError("Extractor has no outputs?")

        self.in_index = self.inp[0]["index"]
        self.out_index = self.out[0]["index"]
        self.in_dtype = self.inp[0]["dtype"]
        self.in_shape = tuple(self.inp[0]["shape"])
        self.eps = float(eps)

    def build_input_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        windows: (N,T) float32 normalized
        returns X: (N,1,T) or (N,1,T,C) depending on model input.
        """
        if windows.ndim != 2:
            raise ValueError(f"windows must be (N,T), got {windows.shape}")

        N, T = int(windows.shape[0]), int(windows.shape[1])

        if len(self.in_shape) == 2:
            # (1,T)
            if self.in_shape[1] != T:
                raise ValueError(f"Extractor expects T={self.in_shape[1]} but got {T}")
            X = windows.reshape(N, 1, T).astype(self.in_dtype, copy=False)
            return X

        if len(self.in_shape) == 3:
            # (1,T,C)
            if self.in_shape[1] != T:
                raise ValueError(f"Extractor expects T={self.in_shape[1]} but got {T}")
            C = int(self.in_shape[2])
            X = np.zeros((N, 1, T, C), dtype=self.in_dtype)
            X[:, 0, :, 0] = windows
            if C > 1:
                for c in range(1, C):
                    X[:, 0, :, c] = windows
            return X

        raise RuntimeError(f"Unsupported extractor input shape: {self.in_shape}")

    def extract_embeddings(self, X: np.ndarray, *, return_mask: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        X: (N,1,T) or (N,1,T,C)
        Runs per-sample inference (safe for TFLite).
        Returns:
          embeddings: (M,D) float32 (M <= N if some samples skipped)
          mask: (N,) bool (True if sample produced embedding) if return_mask=True
        """
        N = int(X.shape[0])

        # Determine output dimension robustly
        # Usually out shape is (1,D); flatten anyway.
        out_shape = tuple(self.out[0]["shape"])
        D = int(np.prod(out_shape)) if len(out_shape) >= 1 else 0
        if D <= 0:
            # fallback: run one sample
            self.interp.set_tensor(self.in_index, X[0].astype(self.in_dtype, copy=False))
            self.interp.invoke()
            D = int(np.asarray(self.interp.get_tensor(self.out_index)).size)

        embs = np.zeros((N, D), dtype=np.float32)
        ok = np.zeros((N,), dtype=bool)

        for i in range(N):
            try:
                xi = X[i]
                if not np.all(np.isfinite(xi)):
                    continue

                self.interp.set_tensor(self.in_index, xi.astype(self.in_dtype, copy=False))
                self.interp.invoke()
                emb = np.asarray(self.interp.get_tensor(self.out_index), dtype=np.float32).reshape(-1)

                if np.linalg.norm(emb) < 1e-9:
                    emb = emb + self.eps

                embs[i, :] = emb
                ok[i] = True
            except Exception:
                continue

        if return_mask:
            return embs[ok], ok
        return embs[ok], None


# -----------------------
# Main pipeline: windows -> labels -> embeddings -> payload
# -----------------------

def build_embeddings_training_payload(
    *,
    fullSensorHistoryData: List[Dict[str, Any]],
    selectedWindows: Optional[List[Dict[str, Any]]] = None,
    applianceSensorHistoryData: Optional[List[Dict[str, Any]]] = None,
    appliance_name: str,
    appliance_type: str = "",
    supervision_mode: str = "intervals",
    appliance_sensor_id: Optional[str] = None,
    inference_dir: str = "/app/inference",
    embeddings_only: bool = True,
    num_threads: int = 2,
    batch_size: int = 1024,
    align_grid: str = "start",
    max_hold_factor: float = 5.0,
    max_hold_s: Optional[float] = DEFAULT_SENSOR_MAX_GAP_S,
    on_threshold_w: float = 20.0,
    min_on_s: float = 60.0,
    min_off_s: float = 300.0,
) -> Dict[str, Any]:
    """
    End-to-end training prep:
      - resample mains to dt
      - build windows in chunks
      - label ON/OFF at pred_idx
      - compute extractor embeddings locally
      - return JSON-safe payload ready to POST to the training server

    No raw mains are returned (unless embeddings_only=False for debugging).
    """
    # Load settings and the local embedding extractor
    settings_path = os.path.join(inference_dir, "model_settings.json")
    extractor_path = os.path.join(inference_dir, "extractor.tflite")

    settings = load_model_settings(settings_path)
    extractor = QueryExtractor(extractor_path, num_threads=num_threads)

    mode = str(supervision_mode or "intervals").strip().lower()
    if mode not in ("intervals", "sensor"):
        raise ValueError(f"Unsupported supervision_mode: {supervision_mode}")

    # Parse mains + supervision source
    pts = parse_mains_points(fullSensorHistoryData)
    if len(pts) < 2:
        raise ValueError("Not enough valid mains points after cleaning.")

    selectedWindows = selectedWindows or []
    applianceSensorHistoryData = applianceSensorHistoryData or []
    on_intervals = normalize_on_intervals(selectedWindows) if mode == "intervals" else []
    gt_pts = parse_power_points(applianceSensorHistoryData) if mode == "sensor" else []

    T = settings.sequence_length
    dt = settings.frequency_s
    pred_idx = settings.pred_idx

    # Build uniform grid across available history
    t_start = pts[0][0]
    t_end = pts[-1][0]
    grid_t = build_uniform_grid(t_start, t_end, dt, align=align_grid)
    if grid_t.size < T:
        required_span_s = max(0.0, (T - 1) * dt)
        raise ValueError(
            "Not enough history for one training window: "
            f"have {t_end - t_start:.1f}s ({grid_t.size} resampled points), "
            f"need at least {required_span_s:.1f}s ({T} points at {dt:.1f}s)."
        )

    # Historical Home Assistant data is a state-change log: an absent row means
    # the last reported value is still current.  An explicit None therefore
    # preserves that state through the selected historical range.  Callers that
    # need a stale-data cutoff can still provide a positive max_hold_s value.
    if max_hold_s is None:
        effective_max_hold_s = None
    elif max_hold_s > 0:
        effective_max_hold_s = float(max_hold_s)
    else:
        effective_max_hold_s = float(max_hold_factor) * dt if max_hold_factor and max_hold_factor > 0 else None
    fill_value_w = float(settings.query_mean)  # best default in the query normalization space

    y_grid, mains_valid_mask = zoh_resample_to_grid(
        pts,
        grid_t,
        max_hold_s=effective_max_hold_s,
        fill_value=fill_value_w,
        return_valid_mask=True,
    )
    mains_energy_wh = float(np.maximum(y_grid, 0.0).sum() * dt / 3600.0) if y_grid.size else 0.0
    mains_mean_w = float(np.mean(y_grid)) if y_grid.size else 0.0

    # Build candidate windows as a lightweight view; we materialize only one chunk at a time.
    M = int(grid_t.size)
    N = M - (T - 1)
    windows_raw = np.lib.stride_tricks.sliding_window_view(y_grid.astype(np.float32), window_shape=T)  # (N,T)
    query_std = settings.query_std if abs(settings.query_std) > 1e-12 else 1.0
    valid_windows_mask = np.all(
        np.lib.stride_tricks.sliding_window_view(mains_valid_mask.astype(np.uint8), window_shape=T) == 1,
        axis=1,
    )

    # Label time for each window at pred_idx:
    # end times are grid_t[T-1:], label time = end - (T-1-pred_idx)*dt
    grid_t_end = grid_t[(T - 1):]
    offset = (T - 1 - pred_idx) * dt
    grid_t_label = grid_t_end - offset
    mains_at_label = y_grid[pred_idx:(pred_idx + N)].astype(np.float32)

    # Create labels
    y_power: Optional[np.ndarray] = None
    y_on = np.zeros((N,), dtype=np.uint8)
    if mode == "intervals":
        if not on_intervals:
            raise ValueError("selectedWindows must contain at least one valid ON interval for interval supervision.")
        idx_ptr = 0
        for k in range(N):
            y_on[k], idx_ptr = _is_on_at_time(float(grid_t_label[k]), on_intervals, idx_ptr)
    else:
        if len(gt_pts) < 2:
            raise ValueError("applianceSensorHistoryData must contain at least two valid points for sensor supervision.")
        gt_grid = zoh_resample_to_grid(gt_pts, grid_t, max_hold_s=effective_max_hold_s, fill_value=0.0)
        gt_grid = np.maximum(gt_grid.astype(np.float32), 0.0)
        y_power = gt_grid[pred_idx:(pred_idx + N)].astype(np.float32)
        gt_on_mask = compute_on_mask(
            gt_grid,
            sample_period_s=dt,
            threshold_watts=float(on_threshold_w),
            window_hours=24.0,
            min_on_s=float(min_on_s),
            min_off_s=float(min_off_s),
        )
        y_on = gt_on_mask[pred_idx:(pred_idx + N)].astype(np.uint8)

    n_windows_after_gap_filter = int(np.count_nonzero(valid_windows_mask))
    if n_windows_after_gap_filter == 0:
        validity = summarize_mains_validity(
            pts,
            mains_valid_mask,
            sample_period_s=dt,
            max_hold_s=effective_max_hold_s,
        )
        required_span_s = max(0.0, (T - 1) * dt)
        stale_detail = (
            f" Found {int(validity['stale_gap_count'])} raw gap(s) above {effective_max_hold_s:.1f}s "
            f"(largest {validity['max_stale_gap_s']:.1f}s)."
            if effective_max_hold_s is not None and validity["stale_gap_count"]
            else ""
        )
        raise ValueError(
            "No valid training windows remain after excluding stale mains samples. "
            f"This model needs {required_span_s:.1f}s of continuous history; the longest valid span is "
            f"{validity['longest_valid_span_s']:.1f}s."
            f"{stale_detail} Choose a range containing a continuous span of that length, "
            "or increase sensor_max_gap_s only when the sensor can reliably hold its last value."
        )

    # Build extractor inputs and compute embeddings chunk-by-chunk to keep peak RAM bounded.
    effective_batch_size = max(1, int(batch_size))
    embeddings_list: List[List[float]] = []
    targets_on_list: List[int] = []
    t_label_list: List[float] = []
    t_end_list: List[float] = []
    targets_power_list: Optional[List[float]] = [] if y_power is not None else None
    weak_mains_list: Optional[List[float]] = [] if y_power is None else None
    selected_mains_energy_wh = 0.0
    appliance_energy_wh = 0.0

    for start in range(0, N, effective_batch_size):
        end = min(N, start + effective_batch_size)
        batch_valid = valid_windows_mask[start:end]
        if not np.any(batch_valid):
            continue

        batch_windows_raw = np.asarray(windows_raw[start:end], dtype=np.float32)
        batch_baseload = np.min(batch_windows_raw, axis=1).astype(np.float32)
        batch_windows = ((batch_windows_raw - batch_baseload[:, None] - float(settings.query_mean)) / float(query_std)).astype(np.float32)
        batch_windows = batch_windows[batch_valid]
        if batch_windows.shape[0] == 0:
            continue

        batch_y_on = y_on[start:end][batch_valid].astype(np.uint8, copy=False)
        batch_t_label = grid_t_label[start:end][batch_valid].astype(np.float64, copy=False)
        batch_t_end = grid_t_end[start:end][batch_valid].astype(np.float64, copy=False)
        batch_mains_at_label = mains_at_label[start:end][batch_valid].astype(np.float32, copy=False)
        batch_weak_mains = np.maximum(batch_mains_at_label - batch_baseload[batch_valid], 0.0).astype(np.float32, copy=False)
        batch_y_power = None
        if y_power is not None:
            batch_y_power = y_power[start:end][batch_valid].astype(np.float32, copy=False)

        X = extractor.build_input_batch(batch_windows)
        embs, mask = extractor.extract_embeddings(X, return_mask=True)
        if mask is None:
            raise RuntimeError("Internal error: mask not returned")
        if embs.shape[0] == 0:
            continue

        batch_y_on_f = batch_y_on[mask].astype(np.uint8, copy=False)
        batch_t_label_f = batch_t_label[mask].astype(np.float64, copy=False)
        batch_t_end_f = batch_t_end[mask].astype(np.float64, copy=False)
        batch_mains_at_label_f = batch_mains_at_label[mask].astype(np.float32, copy=False)

        embeddings_list.extend(embs.tolist())
        targets_on_list.extend(batch_y_on_f.tolist())
        t_label_list.extend(batch_t_label_f.tolist())
        t_end_list.extend(batch_t_end_f.tolist())

        if batch_y_power is not None and targets_power_list is not None:
            batch_y_power_f = batch_y_power[mask].astype(np.float32, copy=False)
            targets_power_list.extend(batch_y_power_f.tolist())
            appliance_energy_wh += float(np.maximum(batch_y_power_f, 0.0).sum() * dt / 3600.0)
        else:
            selected_mains_energy_wh += float(batch_mains_at_label_f[batch_y_on_f == 1].sum() * dt / 3600.0)
            if weak_mains_list is not None:
                batch_weak_mains_f = batch_weak_mains[mask].astype(np.float32, copy=False)
                weak_mains_list.extend(batch_weak_mains_f.tolist())

    if not embeddings_list:
        raise ValueError(
            "No training embeddings could be extracted from the selected range. "
            "Please try a cleaner range or a smaller time span."
        )

    payload: Dict[str, Any] = {
        "appliance_name": appliance_name,
        "appliance_type": appliance_type,
        "supervision_mode": mode,
        "appliance_sensor_id": appliance_sensor_id,
        "settings": {
            "version": settings.version,
            "saved_at": settings.saved_at,
            "sequence_length": settings.sequence_length,
            "frequency": settings.frequency_s,
            "pred_idx": settings.pred_idx,
            "query_mean": settings.query_mean,
            "query_std": settings.query_std,
            "ref_mean": settings.ref_mean,
            "ref_std": settings.ref_std,
            "power_mean": settings.power_mean,
            "power_std": settings.power_std,
        },
        "embeddings": embeddings_list,
        "targets_on": targets_on_list,
        "weak_mains": weak_mains_list,
        "t_label": t_label_list,
        "t_end": t_end_list,
        "stats": {
            "n_raw_points": int(len(fullSensorHistoryData)),
            "n_clean_points": int(len(pts)),
            "grid_points": int(grid_t.size),
            "n_windows_total": int(N),
            "n_windows_after_gap_filter": n_windows_after_gap_filter,
            "n_windows_dropped_mains_gaps": int(N - n_windows_after_gap_filter),
            "n_embeddings_ok": len(embeddings_list),
            "supervision_mode": mode,
            "appliance_sensor_id": appliance_sensor_id,
            "n_on_intervals": int(len(on_intervals)),
            "n_ground_truth_points": int(len(gt_pts)),
            "n_weak_mains": len(weak_mains_list) if weak_mains_list is not None else 0,
            "on_fraction": (float(sum(targets_on_list)) / float(len(targets_on_list))) if targets_on_list else 0.0,
            "mains_mean_w": mains_mean_w,
            "mains_energy_wh": mains_energy_wh,
            "range_start": float(t_start),
            "range_end": float(t_end),
            "range_duration_h": float(max(0.0, t_end - t_start) / 3600.0),
            "align_grid": align_grid,
            "max_hold_s": effective_max_hold_s,
            "fill_value_w": fill_value_w,
            "on_threshold_w": float(on_threshold_w),
            "min_on_s": float(min_on_s),
            "min_off_s": float(min_off_s),
            "batch_size": effective_batch_size,
        },
    }

    if targets_power_list is not None:
        payload["targets_power"] = targets_power_list
        payload["stats"]["power_mean_w"] = (float(sum(targets_power_list)) / float(len(targets_power_list))) if targets_power_list else 0.0
        payload["stats"]["appliance_energy_wh"] = appliance_energy_wh
        payload["stats"]["mains_share_pct"] = float((appliance_energy_wh / mains_energy_wh) * 100.0) if mains_energy_wh > 1e-9 else 0.0
    else:
        payload["stats"]["selected_mains_energy_wh"] = selected_mains_energy_wh
        payload["stats"]["mains_share_pct"] = float((selected_mains_energy_wh / mains_energy_wh) * 100.0) if mains_energy_wh > 1e-9 else 0.0
        payload["stats"]["weak_mains_mean_w"] = (float(sum(weak_mains_list)) / float(len(weak_mains_list))) if weak_mains_list else 0.0

    if not embeddings_only:
        # Debug only — DO NOT send to the training server in production
        payload["debug"] = {
            "grid_t": grid_t.tolist(),
            "y_grid": y_grid.astype(float).tolist(),
        }
        if y_power is not None:
            payload["debug"]["targets_power"] = y_power.astype(float).tolist()

    return payload
