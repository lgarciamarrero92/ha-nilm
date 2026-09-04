from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from embedding_store import is_embedding_marked_deleted, load_embedding_metadata
from runtime_settings import DEFAULT_SENSOR_MAX_GAP_S

try:
    # Preferred for HA app (lightweight)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for dev machines with full TF installed
    from tensorflow.lite import Interpreter


TimeLike = Union[float, int, datetime]
HistoryFetcher = Callable[[datetime, datetime], Awaitable[List[Tuple[float, float]]]]


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


def _to_epoch_seconds(now: TimeLike) -> float:
    if isinstance(now, (int, float)):
        return float(now)
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.timestamp()
    raise TypeError(f"Unsupported time type: {type(now)}")


def _to_utc_datetime(now: TimeLike) -> datetime:
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(now), tz=timezone.utc)


class _RingBuffer:
    """
    Fixed-size ring buffer for float32 values.
    Provides an ordered view (oldest -> newest) when full.
    """
    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros((self.size,), dtype=np.float32)
        self.write_idx = 0
        self.count = 0

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.write_idx = 0
        self.count = 0

    def push(self, x: float) -> None:
        self.buf[self.write_idx] = np.float32(x)
        self.write_idx = (self.write_idx + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def is_full(self) -> bool:
        return self.count >= self.size

    def to_ordered_full(self) -> np.ndarray:
        """
        Return contiguous copy (oldest->newest). Requires full buffer.
        """
        if not self.is_full():
            raise RuntimeError("RingBuffer not full")
        # Oldest element is at write_idx (next write position)
        return np.concatenate((self.buf[self.write_idx:], self.buf[:self.write_idx]), axis=0)


class RefQueryDisaggregator:
    """
    Online RefQuery disaggregator.

    Pipeline:
      - Extractor: mains window (599 samples at 8s) -> query embedding
      - Head: (ref embedding for an appliance, query embedding) -> regression + classification

    Files expected in /app/inference:
      - extractor.tflite
      - head.tflite
      - model_settings.json
      - embeddings/ (optional shipped defaults)

    User embeddings persist in /data/embeddings.
    On startup, any missing default embedding is copied from /app/inference/embeddings to /data/embeddings.

    Warm start:
      - If provided a history fetcher, it will backfill the 599-point window from HA history
        so you don't need to wait ~80 minutes.
      - The 7-day limits and robust error handling should live in ha_client.fetch_history_points()
        (so we don't duplicate that logic here).
    """

    def __init__(
        self,
        inference_dir: str = "/app/inference",
        embeddings_dir: str = "/data/embeddings",
        num_threads: int = 2,
        history_fetcher: Optional[HistoryFetcher] = None,
        max_gap_factor: float = 5.0,
        max_gap_s: Optional[float] = DEFAULT_SENSOR_MAX_GAP_S,
        warm_start_margin_steps: int = 2,
        warm_retry_cooldown_s: float = 300.0,  # retry warm start at most every 5 minutes if desired
        top_k: Optional[int] = None,           # optional gating: only run head for top-K by cosine similarity
        eps: float = 1e-6,
    ):
        self.inference_dir = inference_dir
        self.embeddings_dir = embeddings_dir
        self.num_threads = int(num_threads)
        self.history_fetcher = history_fetcher
        self.max_gap_factor = float(max_gap_factor)
        self.max_gap_s = float(max_gap_s) if max_gap_s and max_gap_s > 0 else None
        self.warm_start_margin_steps = int(warm_start_margin_steps)
        self.warm_retry_cooldown_s = float(warm_retry_cooldown_s)
        self.top_k = top_k
        self.eps = float(eps)

        # ---- Load settings ----
        settings_path = os.path.join(self.inference_dir, "model_settings.json")
        self.settings = self._load_settings(settings_path)

        # ---- Copy default embeddings (shipped) into persistent location if missing ----
        self._ensure_default_embeddings()

        # ---- Online state ----
        self.window = _RingBuffer(self.settings.sequence_length)

        # resampling schedule (model timestep)
        self.last_raw_t: Optional[float] = None
        self.last_raw_y: Optional[float] = None
        self.next_model_t: Optional[float] = None  # next emission time for model sample

        # warm-start controls
        self._warm_attempted = False
        self._warm_last_attempt_t: Optional[float] = None

        # ---- Load interpreters ----
        self.extractor_path = os.path.join(self.inference_dir, "extractor.tflite")
        self.head_path = os.path.join(self.inference_dir, "head.tflite")

        if not os.path.exists(self.extractor_path):
            raise FileNotFoundError(f"Extractor model not found: {self.extractor_path}")
        if not os.path.exists(self.head_path):
            raise FileNotFoundError(f"Head model not found: {self.head_path}")

        self.extractor = Interpreter(model_path=self.extractor_path, num_threads=self.num_threads)
        self.extractor.allocate_tensors()
        self.ex_in = self.extractor.get_input_details()
        self.ex_out = self.extractor.get_output_details()

        if len(self.ex_in) != 1:
            raise RuntimeError(f"Extractor expected 1 input, got {len(self.ex_in)}")
        if len(self.ex_out) < 1:
            raise RuntimeError("Extractor has no outputs?")

        self.head = Interpreter(model_path=self.head_path, num_threads=self.num_threads)
        self.head.allocate_tensors()
        self.hd_in = self.head.get_input_details()
        self.hd_out = self.head.get_output_details()

        if len(self.hd_in) != 2:
            raise RuntimeError(f"Head expected 2 inputs, got {len(self.hd_in)}")
        if len(self.hd_out) != 2:
            raise RuntimeError(f"Head expected 2 outputs, got {len(self.hd_out)}")

        # Bind head inputs by tensor name instead of positional order. The exported
        # TFLite model can expose query/ref tensors in a different order from the
        # original Keras model input list.
        self.hd_ref_in = self.hd_in[0]
        self.hd_query_in = self.hd_in[1]
        for detail in self.hd_in:
            in_name = (detail.get("name", "") or "").lower()
            if "ref_embedding" in in_name or in_name.endswith("ref_embedding"):
                self.hd_ref_in = detail
            elif "query_embedding" in in_name or in_name.endswith("query_embedding"):
                self.hd_query_in = detail

        # ---- Preallocate extractor input tensor ----
        self._extractor_input = np.zeros(tuple(self.ex_in[0]["shape"]), dtype=self.ex_in[0]["dtype"])

        # ---- Embedding cache ----
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._appliance_params_cache: Dict[str, Dict[str, float]] = {}

        # ---- Last result ----
        self.last_result: Optional[dict] = None

    # -------------------- settings / files --------------------

    @staticmethod
    def _load_settings(path: str) -> ModelSettings:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return ModelSettings(
            version=int(raw.get("version", 1)),
            saved_at=str(raw.get("saved_at", "")),
            sequence_length=int(raw["sequence_length"]),
            frequency_s=float(raw["frequency"]),
            pred_idx=int(raw.get("pred_idx", int(raw["sequence_length"]) - 1)),
            query_mean=float(raw["query_mean"]),
            query_std=float(raw["query_std"]),
            ref_mean=float(raw["ref_mean"]),
            ref_std=float(raw["ref_std"]),
            power_mean=float(raw["power_mean"]),
            power_std=float(raw["power_std"]),
        )

    def _ensure_default_embeddings(self) -> None:
        """
        If you ship default embeddings in /app/inference/embeddings,
        copy them into /data/embeddings if missing (do not overwrite user data).
        """
        src_dir = os.path.join(self.inference_dir, "embeddings")
        dst_dir = self.embeddings_dir

        if not os.path.isdir(src_dir):
            # No shipped defaults; fine.
            os.makedirs(dst_dir, exist_ok=True)
            return

        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            if not fname.endswith(".npy"):
                continue
            if is_embedding_marked_deleted(dst_dir, fname[:-4]):
                continue
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    # Do not fail app startup if copying defaults fails.
                    pass

    # -------------------- embeddings --------------------

    def _list_embedding_names(self) -> List[str]:
        if not os.path.isdir(self.embeddings_dir):
            return []
        names = []
        for f in os.listdir(self.embeddings_dir):
            if f.endswith(".npy"):
                names.append(f[:-4])
        names.sort()
        return names

    def _load_embedding(self, appliance_name: str) -> np.ndarray:
        """
        Load appliance embedding from /data/embeddings/<name>.npy and cache it.
        """
        if appliance_name in self._emb_cache:
            return self._emb_cache[appliance_name]

        path = os.path.join(self.embeddings_dir, f"{appliance_name}.npy")
        emb = np.load(path).astype(np.float32).reshape(1, -1)

        # avoid pathological all-zero
        if np.linalg.norm(emb) < 1e-9:
            emb = emb + self.eps

        self._emb_cache[appliance_name] = emb
        return emb

    def _load_appliance_params(self, appliance_name: str) -> Dict[str, float]:
        if appliance_name in self._appliance_params_cache:
            return self._appliance_params_cache[appliance_name]

        metadata = load_embedding_metadata(self.embeddings_dir, appliance_name) or {}
        params = {
            "onoff_threshold": float(metadata.get("onoff_threshold", 0.5)),
            "power_threshold": float(metadata.get("power_threshold", 0.0)),
        }
        self._appliance_params_cache[appliance_name] = params
        return params

    # -------------------- normalization --------------------

    def _normalize_mains(self, x: float) -> float:
        std = self.settings.query_std if abs(self.settings.query_std) > 1e-12 else 1.0
        return (x - self.settings.query_mean) / std

    def _denormalize_power(self, y_norm: float) -> float:
        return y_norm * self.settings.power_std + self.settings.power_mean

    def _prediction_delay_s(self) -> float:
        return float(max(0, self.settings.sequence_length - 1 - self.settings.pred_idx)) * float(self.settings.frequency_s)

    # -------------------- warm state --------------------

    def is_warm(self) -> bool:
        return self.window.is_full()

    def reset(self) -> None:
        """
        Reset internal window and schedule. Next calls will rebuild the window online.
        Warm-start may be attempted again (subject to cooldown).
        """
        self.window.reset()
        self.last_raw_t = None
        self.last_raw_y = None
        self.next_model_t = None
        self.last_result = None
        # allow warm-start again
        self._warm_attempted = False

    # -------------------- online resampling --------------------

    def _emit_model_samples(self, t: float, y: float) -> int:
        """
        Insert 0..N model-rate points into the ring buffer using ZOH.
        Returns number of emitted model samples.
        """
        dt = self.settings.frequency_s

        # first sample initializes schedule
        if self.last_raw_t is None:
            self.last_raw_t = t
            self.last_raw_y = y
            self.next_model_t = t  # emit immediately

        # handle long gaps robustly: reset schedule/window (self-heal)
        assert self.last_raw_t is not None
        gap = t - self.last_raw_t
        max_gap_s = self.max_gap_s if self.max_gap_s is not None else self.max_gap_factor * dt
        if gap > max_gap_s:
            # large gap: treat as discontinuity
            self.reset()
            # re-init
            self.last_raw_t = t
            self.last_raw_y = y
            self.next_model_t = t

        emitted = 0
        assert self.next_model_t is not None

        # ZOH resampling: until current time, emit values using last_raw_y
        while self.next_model_t <= t + 1e-9:
            use_val = self.last_raw_y if self.last_raw_y is not None else y
            self.window.push(float(use_val))
            self.next_model_t += dt
            emitted += 1

        # update last raw sample
        self.last_raw_t = t
        self.last_raw_y = y
        return emitted

    # -------------------- inference helpers --------------------

    def _build_extractor_input(self) -> np.ndarray:
        """
        Fill preallocated extractor input tensor from ring buffer (oldest->newest).
        Supports input shapes [1,T] and [1,T,1]/[1,T,C].
        """
        ordered = self.window.to_ordered_full().astype(np.float32, copy=True)
        T = self.settings.sequence_length
        ordered = ordered - np.min(ordered)
        ordered = np.asarray([self._normalize_mains(float(v)) for v in ordered], dtype=np.float32)

        shape = tuple(self.ex_in[0]["shape"])

        if len(shape) == 2:
            # [1, T]
            if shape[1] != T:
                raise RuntimeError(f"Extractor expects T={shape[1]} but settings has {T}")
            self._extractor_input[0, :] = ordered
        elif len(shape) == 3:
            # [1, T, C]
            if shape[1] != T:
                raise RuntimeError(f"Extractor expects T={shape[1]} but settings has {T}")
            self._extractor_input[0, :, 0] = ordered
            if shape[2] > 1:
                for c in range(1, shape[2]):
                    self._extractor_input[0, :, c] = ordered
        else:
            raise RuntimeError(f"Unsupported extractor input shape: {shape}")

        return self._extractor_input

    @staticmethod
    def _to_probability(value: float) -> float:
        if value < -0.05 or value > 1.05:
            value = 1.0 / (1.0 + np.exp(-value))
        return float(np.clip(value, 0.0, 1.0))

    def _extract_query_embedding(self) -> np.ndarray:
        x = self._build_extractor_input()
        self.extractor.set_tensor(self.ex_in[0]["index"], x)
        self.extractor.invoke()

        emb = self.extractor.get_tensor(self.ex_out[0]["index"])
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        if np.linalg.norm(emb) < 1e-9:
            emb = emb + self.eps

        return emb

    def _run_head(self, ref_emb: np.ndarray, query_emb: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (power_watts, onoff, power_norm)
        """
        self.head.set_tensor(
            self.hd_ref_in["index"],
            ref_emb.astype(self.hd_ref_in["dtype"], copy=False),
        )
        self.head.set_tensor(
            self.hd_query_in["index"],
            query_emb.astype(self.hd_query_in["dtype"], copy=False),
        )
        self.head.invoke()

        out0 = float(np.ravel(self.head.get_tensor(self.hd_out[0]["index"]))[0])
        out1 = float(np.ravel(self.head.get_tensor(self.hd_out[1]["index"]))[0])

        n0 = (self.hd_out[0].get("name", "") or "").lower()
        n1 = (self.hd_out[1].get("name", "") or "").lower()

        if ("power" in n0) or ("regression" in n0):
            power_norm, onoff = out0, out1
        elif ("power" in n1) or ("regression" in n1):
            power_norm, onoff = out1, out0
        else:
            # fallback ordering you observed
            power_norm, onoff = out0, out1

        power_w = self._denormalize_power(power_norm)
        onoff_prob = self._to_probability(onoff)
        return power_w, onoff_prob, power_norm

    # -------------------- optional gating (top-K) --------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
        # a,b are (1,D)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return -1.0
        return float(np.dot(a.ravel(), b.ravel()) / (na * nb))

    def _select_appliances(self, query_emb: np.ndarray, names: List[str]) -> List[str]:
        """
        If top_k is set, choose top_k appliances by cosine similarity to speed up.
        Otherwise return all.
        """
        if not self.top_k or self.top_k <= 0 or self.top_k >= len(names):
            return names

        scored = []
        for name in names:
            try:
                ref = self._load_embedding(name)
                sim = self._cosine_similarity(ref, query_emb, eps=1e-9)
                scored.append((sim, name))
            except Exception:
                continue

        scored.sort(reverse=True, key=lambda x: x[0])
        return [n for _, n in scored[: self.top_k]]

    # -------------------- warm start logic --------------------

    async def _try_warm_start(self, now_dt: datetime) -> None:
        """
        Try to prefill the 599-window by fetching HA history and resampling at dt.
        This method NEVER raises; it is designed to be safe in the online loop.
        """
        if self.history_fetcher is None:
            return
        if self.is_warm():
            return

        now_epoch = now_dt.timestamp()

        # Cooldown to avoid hammering HA if it fails repeatedly
        if self._warm_last_attempt_t is not None:
            if (now_epoch - self._warm_last_attempt_t) < self.warm_retry_cooldown_s:
                return

        self._warm_last_attempt_t = now_epoch

        dt = self.settings.frequency_s
        lookback_s = self.settings.sequence_length * dt
        # add small margin for safety (a couple of steps)
        start_dt = now_dt - timedelta(seconds=lookback_s + self.warm_start_margin_steps * dt)

        points: List[Tuple[float, float]] = []
        try:
            points = await self.history_fetcher(start_dt, now_dt)
        except Exception:
            # keep running even if HA history fetch fails
            return

        if not points:
            return

        # sanitize + sort
        clean: List[Tuple[float, float]] = []
        for ts, val in points:
            try:
                clean.append((float(ts), float(val)))
            except Exception:
                continue
        clean.sort(key=lambda p: p[0])
        if not clean:
            return

        # Build model-time grid ending at now (uniform dt, seq_len points)
        # We'll use ZOH across the raw history points.
        T = self.settings.sequence_length
        t0 = now_epoch - (T - 1) * dt
        grid = [t0 + i * dt for i in range(T)]

        j = 0
        last_val = clean[0][1]

        # Fill ring buffer from scratch
        self.window.reset()
        for tg in grid:
            while j < len(clean) and clean[j][0] <= tg:
                last_val = clean[j][1]
                j += 1
            self.window.push(float(last_val))

        # Anchor to now so that the first live _emit_model_samples call sees
        # gap ≈ 0 and does not trigger a max_gap reset that would undo the
        # warm start. Using clean[-1][0] (the last history timestamp) would
        # cause gap = now - history_ts >> max_gap_s on the very next call.
        self.last_raw_t = now_epoch
        self.last_raw_y = clean[-1][1]
        self.next_model_t = now_epoch + dt

        # Mark that we have attempted warm start successfully (window full now)
        self._warm_attempted = True

    # -------------------- main API --------------------

    async def disaggregate_next(
        self,
        total_power: float,
        now: TimeLike,
        appliances: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """
        Call this for every new mains reading.

        Returns None if:
          - no new model timestep emitted yet, OR
          - window not warm/full yet.

        Otherwise returns:
          {
            "timestamp": <epoch seconds of prediction target>,
            "window_end_timestamp": <epoch seconds of newest model sample in the window>,
            "raw_timestamp": <epoch seconds of the raw sample that triggered inference>,
            "prediction_delay_s": <seconds between window end and prediction target>,
            "model_dt_s": <frequency>,
            "appliances": {
               "<name>": {"power": float, "onoff": float, "power_norm": float},
               ...
            }
          }
        """
        t = _to_epoch_seconds(now)
        now_dt = _to_utc_datetime(now)

        # If not warm, try to warm-start from HA history once (or with cooldown)
        if (not self.is_warm()) and (self.history_fetcher is not None):
            # We do not permanently forbid later attempts unless you want that;
            # cooldown prevents spamming HA.
            await self._try_warm_start(now_dt)

        # Feed live sample into resampler/ring buffer
        try:
            emitted = self._emit_model_samples(t, float(total_power))
        except Exception:
            # If something goes very wrong, reset and continue safely.
            self.reset()
            return None

        # If no model-rate sample emitted, nothing to do (robust under fast WS updates)
        if emitted <= 0:
            return None

        # Need full window to run inference
        if not self.is_warm():
            return None

        window_end_t = float(self.next_model_t - self.settings.frequency_s) if self.next_model_t is not None else float(t)
        prediction_delay_s = self._prediction_delay_s()
        target_t = window_end_t - prediction_delay_s

        # Run inference once per model step
        try:
            query_emb = self._extract_query_embedding()
        except Exception:
            # If the model fails, reset window (safer) and try again later
            self.reset()
            return None

        # Use the same per-window baseline logic as the query path so appliance
        # predictions are capped against the non-baseload portion of mains.
        raw_window = self.window.to_ordered_full()
        window_min = float(np.min(raw_window))
        available_appliance_power = float(max(0.0, float(total_power) - window_min))

        # Determine appliance list
        names = appliances if appliances is not None else self._list_embedding_names()
        if not names:
            return {
                "timestamp": target_t,
                "window_end_timestamp": window_end_t,
                "raw_timestamp": t,
                "prediction_delay_s": prediction_delay_s,
                "model_dt_s": self.settings.frequency_s,
                "pred_idx": self.settings.pred_idx,
                "appliances": {},
            }

        # Optional top-K gating for efficiency if you have many appliances
        names = self._select_appliances(query_emb, names)

        out: Dict[str, dict] = {}

        for name in names:
            try:
                ref = self._load_embedding(name)
                params = self._load_appliance_params(name)
                # embedding dimension mismatch check
                if ref.shape[1] != query_emb.shape[1]:
                    continue
                power_w, onoff, power_norm = self._run_head(ref, query_emb)
                power_w = float(max(0.0, power_w))
                power_w = float(min(power_w, available_appliance_power))
                onoff_threshold = float(params.get("onoff_threshold", 0.5))
                if onoff < onoff_threshold:
                    power_w = 0.0
                out[name] = {
                    "power": power_w,
                    "onoff": onoff,
                    "power_norm": power_norm,
                    "onoff_threshold": onoff_threshold,
                }
            except FileNotFoundError:
                # missing embedding -> skip
                continue
            except Exception:
                # keep going; one bad appliance shouldn't kill all
                continue

        result = {
            "timestamp": target_t,
            "window_end_timestamp": window_end_t,
            "raw_timestamp": t,
            "prediction_delay_s": prediction_delay_s,
            "model_dt_s": self.settings.frequency_s,
            "pred_idx": self.settings.pred_idx,
            "appliances": out,
        }
        self.last_result = result
        return result
