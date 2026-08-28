from __future__ import annotations

import json
import os
import sys
import time
import gc
import tempfile
from collections import defaultdict

import numpy as np

from prepare_training_data import (
    QueryExtractor,
    build_uniform_grid,
    load_model_settings,
    parse_mains_points,
    zoh_resample_to_grid,
)
from refquery import RefQueryDisaggregator
from runtime_settings import DEFAULT_SENSOR_MAX_GAP_S, clamp_sensor_max_gap_s


PREVIEW_BATCH_SIZE = 1024


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def allocate_preview_power_budget(candidates, available_power_w):
    """Scale active model estimates so their total fits one mains feed."""
    total_candidate_power = sum(max(0.0, float(candidate[1])) for candidate in candidates)
    available_power_w = max(0.0, float(available_power_w))
    scale = min(1.0, available_power_w / total_candidate_power) if total_candidate_power > 0 else 0.0
    return [
        (safe_name, max(0.0, float(candidate_power_w)) * scale, onoff_value, onoff_threshold)
        for safe_name, candidate_power_w, onoff_value, onoff_threshold in candidates
    ]


def summarize_state_series(state_series):
    series = list(state_series or [])
    if not series:
        return {
            "n_points": 0,
            "max_score": None,
            "min_score": None,
            "mean_score": None,
            "threshold": None,
            "n_above_threshold": 0,
        }

    scores = []
    thresholds = []
    n_above = 0
    for point in series:
        score = point.get("score")
        threshold = point.get("threshold")
        if isinstance(score, (int, float)) and np.isfinite(score):
            scores.append(float(score))
        if isinstance(threshold, (int, float)) and np.isfinite(threshold):
            thresholds.append(float(threshold))
            if isinstance(score, (int, float)) and np.isfinite(score) and float(score) >= float(threshold):
                n_above += 1

    if not scores:
        return {
            "n_points": len(series),
            "max_score": None,
            "min_score": None,
            "mean_score": None,
            "threshold": float(thresholds[0]) if thresholds else None,
            "n_above_threshold": n_above,
        }

    return {
        "n_points": len(series),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
        "threshold": float(thresholds[0]) if thresholds else None,
        "n_above_threshold": int(n_above),
    }


def build_offline_preview_context(
    points,
    inference_dir: str,
    num_threads: int = 2,
    align_grid: str = "start",
    max_hold_factor: float = 5.0,
    max_hold_s: float = DEFAULT_SENSOR_MAX_GAP_S,
):
    parsed_points = parse_mains_points([{"x": float(ts) * 1000.0, "y": float(value)} for ts, value in (points or [])])
    if len(parsed_points) < 2:
        return {
            "sequence_length": 0,
            "pred_idx": 0,
            "dt": 0.0,
            "query_mean": 0.0,
            "query_std": 1.0,
            "y_grid": np.zeros((0,), dtype=np.float32),
            "mains_valid_mask": np.zeros((0,), dtype=np.uint8),
            "grid_t": np.zeros((0,), dtype=np.float64),
            "inference_dir": inference_dir,
            "num_threads": int(num_threads),
        }

    settings = load_model_settings(os.path.join(inference_dir, "model_settings.json"))
    T = settings.sequence_length
    dt = settings.frequency_s
    pred_idx = settings.pred_idx
    t_start = parsed_points[0][0]
    t_end = parsed_points[-1][0]
    grid_t = build_uniform_grid(t_start, t_end, dt, align=align_grid)
    if grid_t.size < T:
        return {
            "sequence_length": int(T),
            "pred_idx": int(pred_idx),
            "dt": float(dt),
            "query_mean": float(settings.query_mean),
            "query_std": float(settings.query_std if abs(settings.query_std) > 1e-12 else 1.0),
            "y_grid": np.zeros((0,), dtype=np.float32),
            "mains_valid_mask": np.zeros((0,), dtype=np.uint8),
            "grid_t": np.zeros((0,), dtype=np.float64),
            "inference_dir": inference_dir,
            "num_threads": int(num_threads),
        }

    max_hold_s = float(max_hold_s) if max_hold_s and max_hold_s > 0 else (
        float(max_hold_factor) * dt if max_hold_factor and max_hold_factor > 0 else None
    )
    fill_value_w = float(settings.query_mean)
    y_grid, mains_valid_mask = zoh_resample_to_grid(
        parsed_points,
        grid_t,
        max_hold_s=max_hold_s,
        fill_value=fill_value_w,
        return_valid_mask=True,
    )

    return {
        "sequence_length": int(T),
        "pred_idx": int(pred_idx),
        "dt": float(dt),
        "query_mean": float(settings.query_mean),
        "query_std": float(settings.query_std if abs(settings.query_std) > 1e-12 else 1.0),
        "y_grid": y_grid.astype(np.float32, copy=False),
        "mains_valid_mask": mains_valid_mask.astype(np.uint8, copy=False),
        "grid_t": grid_t.astype(np.float64, copy=False),
        "inference_dir": inference_dir,
        "num_threads": int(num_threads),
    }


def iter_preview_batches(preview_context, batch_size: int = PREVIEW_BATCH_SIZE):
    y_grid = np.asarray(preview_context.get("y_grid"), dtype=np.float32)
    mains_valid_mask = np.asarray(preview_context.get("mains_valid_mask"), dtype=np.uint8)
    grid_t = np.asarray(preview_context.get("grid_t"), dtype=np.float64)
    T = int(preview_context.get("sequence_length") or 0)
    pred_idx = int(preview_context.get("pred_idx") or 0)
    dt = float(preview_context.get("dt") or 0.0)
    query_mean = float(preview_context.get("query_mean") or 0.0)
    query_std = float(preview_context.get("query_std") or 1.0)

    if T <= 0 or y_grid.size < T or grid_t.size < T:
        return

    total_candidates = int(y_grid.size - (T - 1))
    if total_candidates <= 0:
        return

    windows_raw = np.lib.stride_tricks.sliding_window_view(y_grid, window_shape=T)
    valid_windows_mask = np.all(
        np.lib.stride_tricks.sliding_window_view(mains_valid_mask, window_shape=T) == 1,
        axis=1,
    )
    grid_t_end = grid_t[(T - 1):]
    offset = (T - 1 - pred_idx) * dt
    label_times_ms_all = np.round((grid_t_end - offset) * 1000.0).astype(np.int64)
    mains_at_label_all = y_grid[pred_idx:(pred_idx + total_candidates)].astype(np.float32, copy=False)

    effective_batch_size = max(1, int(batch_size))
    for start in range(0, total_candidates, effective_batch_size):
        end = min(total_candidates, start + effective_batch_size)
        batch_valid = valid_windows_mask[start:end]
        if not np.any(batch_valid):
            continue

        batch_windows_raw = np.asarray(windows_raw[start:end], dtype=np.float32)
        batch_baseload = np.min(batch_windows_raw, axis=1).astype(np.float32)
        batch_windows = ((batch_windows_raw - batch_baseload[:, None] - query_mean) / query_std).astype(np.float32)

        yield {
            "windows": batch_windows[batch_valid],
            "label_times_ms": label_times_ms_all[start:end][batch_valid],
            "mains_at_label": mains_at_label_all[start:end][batch_valid],
            "baseload_at_label": batch_baseload[batch_valid],
        }

        del batch_windows_raw
        del batch_baseload
        del batch_windows


def count_preview_points(preview_context):
    y_grid = np.asarray(preview_context.get("y_grid"), dtype=np.float32)
    mains_valid_mask = np.asarray(preview_context.get("mains_valid_mask"), dtype=np.uint8)
    T = int(preview_context.get("sequence_length") or 0)
    if T <= 0 or y_grid.size < T:
        return 0
    valid_windows_mask = np.all(
        np.lib.stride_tricks.sliding_window_view(mains_valid_mask, window_shape=T) == 1,
        axis=1,
    )
    return int(np.count_nonzero(valid_windows_mask))


class PredictionSpool:
    def __init__(self, model_entries):
        self.entries = list(model_entries or [])
        self.paths = {}
        self.handles = {}
        self.first_record = {}
        self.state_stats = {}

        for entry in self.entries:
            safe_name = entry["safe_name"]
            fd, path = tempfile.mkstemp(prefix=f"nilm_preview_{safe_name}_", suffix=".jsonl")
            os.close(fd)
            handle = open(path, "w", encoding="utf-8")
            self.paths[safe_name] = path
            self.handles[safe_name] = handle
            self.first_record[safe_name] = True
            self.state_stats[safe_name] = {
                "n_points": 0,
                "max_score": None,
                "min_score": None,
                "score_sum": 0.0,
                "threshold": None,
                "n_above_threshold": 0,
            }

    def append(self, safe_name, target_ms, power_w, baseload_value, onoff_value, onoff_threshold):
        handle = self.handles[safe_name]
        record = {
            "x": int(target_ms),
            "power": float(power_w),
            "baseload": float(baseload_value),
            "state": int(onoff_value >= onoff_threshold),
            "score": float(onoff_value),
            "threshold": float(onoff_threshold),
        }
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")

        stats = self.state_stats[safe_name]
        score = float(onoff_value)
        threshold = float(onoff_threshold)
        stats["n_points"] += 1
        stats["score_sum"] += score
        stats["max_score"] = score if stats["max_score"] is None else max(float(stats["max_score"]), score)
        stats["min_score"] = score if stats["min_score"] is None else min(float(stats["min_score"]), score)
        if stats["threshold"] is None:
            stats["threshold"] = threshold
        if score >= threshold:
            stats["n_above_threshold"] += 1

    def build_state_summary(self, safe_name):
        stats = self.state_stats.get(safe_name) or {}
        n_points = int(stats.get("n_points") or 0)
        if n_points <= 0:
            return summarize_state_series([])
        return {
            "n_points": n_points,
            "max_score": stats.get("max_score"),
            "min_score": stats.get("min_score"),
            "mean_score": float(stats.get("score_sum", 0.0) / n_points),
            "threshold": stats.get("threshold"),
            "n_above_threshold": int(stats.get("n_above_threshold") or 0),
        }

    def load_model_payload(self, safe_name):
        power_series = []
        baseload_series = []
        state_series = []
        path = self.paths[safe_name]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                point = json.loads(raw)
                x = int(point["x"])
                power_series.append({"x": x, "y": float(point["power"])})
                baseload_series.append({"x": x, "y": float(point["baseload"])})
                state_series.append({
                    "x": x,
                    "y": int(point["state"]),
                    "score": float(point["score"]),
                    "threshold": float(point["threshold"]),
                })
        return {
            "power_series": power_series,
            "baseload_series": baseload_series,
            "state_series": state_series,
            "state_summary": self.build_state_summary(safe_name),
        }

    def close(self):
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles = {}

    def cleanup(self):
        self.close()
        for path in self.paths.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


def write_chunk_payload(model_entries, chunk_predictions_by_model):
    fd, chunk_path = tempfile.mkstemp(prefix="nilm_preview_chunk_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            predictions = []
            for entry in model_entries:
                prediction = chunk_predictions_by_model.get(entry["safe_name"]) or {}
                power_series = prediction.get("power_series") or []
                state_series = prediction.get("state_series") or []
                if not power_series and not state_series:
                    continue
                predictions.append({
                    "model_key": entry["model_key"],
                    "model_name": entry["model_name"],
                    "main_sensor_id": entry.get("main_sensor_id"),
                    "mains_sensor_id": entry.get("mains_sensor_id") or entry.get("main_sensor_id"),
                    "power_series": power_series,
                    "baseload_series": prediction.get("baseload_series") or [],
                    "state_series": state_series,
                })
            json.dump({"predictions": predictions}, f, ensure_ascii=False)
        return chunk_path
    except Exception:
        try:
            os.remove(chunk_path)
        except OSError:
            pass
        raise


def write_result_payload_from_spool(result_path, mode, model_entries, spool):
    if mode == "all":
        with open(result_path, "w", encoding="utf-8") as f:
            f.write('{"predictions":[')
            for index, entry in enumerate(model_entries):
                if index > 0:
                    f.write(",")
                prediction = spool.load_model_payload(entry["safe_name"]) if spool is not None else {
                    "power_series": [],
                    "baseload_series": [],
                    "state_series": [],
                    "state_summary": {},
                }
                json.dump({
                    "model_key": entry["model_key"],
                    "model_name": entry["model_name"],
                    "main_sensor_id": entry.get("main_sensor_id"),
                    "mains_sensor_id": entry.get("mains_sensor_id") or entry.get("main_sensor_id"),
                    "power_series": prediction.get("power_series", []),
                    "baseload_series": prediction.get("baseload_series", []),
                    "state_series": prediction.get("state_series", []),
                    "state_summary": prediction.get("state_summary", {}),
                }, f, ensure_ascii=False)
            f.write("]}")
        return {"prediction_count": len(model_entries)}

    entry = model_entries[0] if model_entries else None
    prediction = spool.load_model_payload(entry["safe_name"]) if (spool is not None and entry is not None) else {
        "power_series": [],
        "baseload_series": [],
        "state_series": [],
        "state_summary": {},
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"result": prediction}, f, ensure_ascii=False)
    state_summary = prediction.get("state_summary") or {}
    return {"n_points": int(state_summary.get("n_points") or 0)}


def _worker_percent(phase, processed, total):
    if total <= 0:
        return 30
    fraction = max(0.0, min(1.0, float(processed) / float(total)))
    if phase == "embeddings":
        return int(round(25 + fraction * 30))
    if phase == "inference":
        return int(round(55 + fraction * 37))
    return int(round(25 + fraction * 67))


def score_predictions_with_progress(
    model_entries,
    preview_context,
    *,
    stream_by_chunk: bool = False,
    keep_full_result: bool = True,
):
    if not model_entries:
        return None

    grouped = defaultdict(list)
    for entry in model_entries:
        grouped[entry["bundle_id"]].append(entry)

    total = count_preview_points(preview_context)
    emit({"phase": "inference", "processed": 0, "total": total, "percent": _worker_percent("inference", 0, total)})

    inference_dir = str(preview_context.get("inference_dir") or "")
    num_threads = int(preview_context.get("num_threads") or 2)
    batch_size = int(preview_context.get("batch_size") or PREVIEW_BATCH_SIZE)
    if total <= 0 or not inference_dir:
        return PredictionSpool(model_entries) if keep_full_result else None

    extractor = QueryExtractor(os.path.join(inference_dir, "extractor.tflite"), num_threads=num_threads)
    last_embedding_emit = 0.0
    last_inference_emit = 0.0
    processed_embeddings = 0
    processed_inference = 0
    emit({"phase": "embeddings", "processed": 0, "total": total, "percent": _worker_percent("embeddings", 0, total)})

    bundle_states = {}
    spool = PredictionSpool(model_entries) if keep_full_result else None
    emitted_chunks = 0
    for bundle_id, entries in grouped.items():
        first = entries[0]
        disaggregator = RefQueryDisaggregator(
            inference_dir=first["inference_dir"],
            embeddings_dir=first["embeddings_dir"],
            num_threads=2,
            history_fetcher=None,
            top_k=None,
        )

        model_state = {}
        for entry in entries:
            ref = disaggregator._load_embedding(entry["safe_name"])
            params = disaggregator._load_appliance_params(entry["safe_name"])
            model_state[entry["safe_name"]] = (ref, float(params.get("onoff_threshold", 0.5)))
        bundle_states[bundle_id] = {
            "disaggregator": disaggregator,
            "model_state": model_state,
        }

    for batch in iter_preview_batches(preview_context, batch_size=batch_size):
        windows = np.asarray(batch.get("windows"), dtype=np.float32)
        if windows.ndim != 2 or windows.shape[0] == 0:
            continue

        label_times_ms = np.asarray(batch.get("label_times_ms"), dtype=np.int64)
        mains_at_label = np.asarray(batch.get("mains_at_label"), dtype=np.float32)
        baseload_at_label = np.asarray(batch.get("baseload_at_label"), dtype=np.float32)

        X = extractor.build_input_batch(windows)
        batch_embeddings = []
        batch_rows = []
        for idx in range(int(X.shape[0])):
            try:
                xi = X[idx]
                if not np.all(np.isfinite(xi)):
                    continue
                extractor.interp.set_tensor(extractor.in_index, xi.astype(extractor.in_dtype, copy=False))
                extractor.interp.invoke()
                emb = np.asarray(extractor.interp.get_tensor(extractor.out_index), dtype=np.float32).reshape(-1)
                if np.linalg.norm(emb) < 1e-9:
                    emb = emb + extractor.eps
                batch_embeddings.append(emb)
                batch_rows.append(idx)
            except Exception:
                continue

            processed_embeddings += 1
            now_mono = time.monotonic()
            if processed_embeddings == 1 or processed_embeddings == total or now_mono - last_embedding_emit >= 0.05:
                emit({
                    "phase": "embeddings",
                    "processed": processed_embeddings,
                    "total": total,
                    "percent": _worker_percent("embeddings", processed_embeddings, total),
                })
                last_embedding_emit = now_mono

        if not batch_embeddings:
            del X
            gc.collect()
            continue

        valid_rows = np.asarray(batch_rows, dtype=np.int64)
        embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        batch_label_times = label_times_ms[valid_rows]
        batch_mains = mains_at_label[valid_rows]
        batch_baseload = baseload_at_label[valid_rows]
        chunk_predictions_by_model = None
        if stream_by_chunk:
            chunk_predictions_by_model = {}
            for entry in model_entries:
                chunk_predictions_by_model[entry["safe_name"]] = {
                    "power_series": [],
                    "baseload_series": [],
                    "state_series": [],
                }

        for idx in range(int(embeddings.shape[0])):
            query_emb = np.asarray(embeddings[idx], dtype=np.float32).reshape(1, -1)
            target_ms = int(batch_label_times[idx])
            baseload_value = float(max(0.0, batch_baseload[idx])) if idx < batch_baseload.size else 0.0
            available_appliance_power = float(max(0.0, float(batch_mains[idx]) - baseload_value)) if idx < batch_mains.size else 0.0

            candidates = []
            for bundle_state in bundle_states.values():
                for safe_name, (ref, onoff_threshold) in bundle_state["model_state"].items():
                    power_w, onoff_value, _power_norm = bundle_state["disaggregator"]._run_head(ref, query_emb)
                    power_w = float(max(0.0, power_w))
                    if onoff_value < onoff_threshold:
                        power_w = 0.0

                    candidates.append((safe_name, power_w, float(onoff_value), float(onoff_threshold)))

            for safe_name, candidate_power_w, onoff_value, onoff_threshold in allocate_preview_power_budget(
                candidates,
                available_appliance_power,
            ):
                # Models assigned to one mains feed share its non-baseload power budget.
                power_w = candidate_power_w

                if spool is not None:
                    spool.append(safe_name, target_ms, power_w, baseload_value, onoff_value, onoff_threshold)
                if stream_by_chunk:
                    chunk_predictions_by_model[safe_name]["power_series"].append({"x": target_ms, "y": power_w})
                    chunk_predictions_by_model[safe_name]["baseload_series"].append({"x": target_ms, "y": baseload_value})
                    chunk_predictions_by_model[safe_name]["state_series"].append({
                        "x": target_ms,
                        "y": int(onoff_value >= onoff_threshold),
                        "score": float(onoff_value),
                        "threshold": float(onoff_threshold),
                    })

            processed_inference += 1
            now_mono = time.monotonic()
            if processed_inference == 1 or processed_inference == total or now_mono - last_inference_emit >= 0.05:
                emit({
                    "phase": "inference",
                    "processed": processed_inference,
                    "total": total,
                    "percent": _worker_percent("inference", processed_inference, total),
                })
                last_inference_emit = now_mono

        if stream_by_chunk:
            chunk_path = write_chunk_payload(model_entries, chunk_predictions_by_model)
            emitted_chunks += 1
            emit({
                "phase": "inference",
                "processed": processed_inference,
                "total": total,
                "percent": _worker_percent("inference", processed_inference, total),
                "chunk_path": chunk_path,
                "chunk_index": emitted_chunks,
            })

        del X
        del batch_embeddings
        del batch_rows
        del valid_rows
        del embeddings
        del batch_label_times
        del batch_mains
        del batch_baseload
        if chunk_predictions_by_model is not None:
            del chunk_predictions_by_model
        gc.collect()

    if spool is not None:
        spool.close()
    return spool


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: edge_preview_worker.py <payload_json>")

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        payload = json.load(f)

    result_path = str(payload.get("result_path") or "").strip()
    points = payload.get("points") or []
    model_entries = payload.get("models") or []
    if not points:
        result_payload = {"predictions": []} if payload.get("mode") == "all" else {"result": {}}
        if result_path:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False)
            emit({"done": True, "result_path": result_path})
        else:
            emit({"done": True, **result_payload})
        return 0

    sensor_max_gap_s = clamp_sensor_max_gap_s(payload.get("sensor_max_gap_s"))
    preview_context = build_offline_preview_context(points, inference_dir=model_entries[0]["inference_dir"], num_threads=2, align_grid="start", max_hold_s=sensor_max_gap_s)
    preview_context["batch_size"] = int(payload.get("batch_size") or PREVIEW_BATCH_SIZE)
    if count_preview_points(preview_context) <= 0:
        result_payload = None
        if payload.get("mode") == "all":
            result_payload = {"predictions": []}
        else:
            result_payload = {"result": {"power_series": [], "baseload_series": [], "state_series": [], "state_summary": {}}}
        if result_path:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, ensure_ascii=False)
            emit({"done": True, "result_path": result_path})
        else:
            emit({"done": True, **(result_payload or {})})
        return 0

    stream_by_chunk = bool(payload.get("stream_chunks")) or payload.get("mode") == "all"
    keep_full_result = payload.get("mode") != "all"
    spool = score_predictions_with_progress(
        model_entries,
        preview_context,
        stream_by_chunk=stream_by_chunk,
        keep_full_result=keep_full_result,
    )
    try:
        if stream_by_chunk and not keep_full_result:
            emit({"done": True, "prediction_count": len(model_entries)})
            return 0
        if result_path:
            meta = write_result_payload_from_spool(result_path, payload.get("mode"), model_entries, spool)
            emit({"done": True, "result_path": result_path, **meta})
            return 0

        result_payload = None
        if payload.get("mode") == "all":
            predictions = []
            for entry in model_entries:
                prediction = spool.load_model_payload(entry["safe_name"]) if spool is not None else {
                    "power_series": [],
                    "baseload_series": [],
                    "state_series": [],
                    "state_summary": {},
                }
                predictions.append({
                    "model_key": entry["model_key"],
                    "model_name": entry["model_name"],
                    "main_sensor_id": entry.get("main_sensor_id"),
                    "mains_sensor_id": entry.get("mains_sensor_id") or entry.get("main_sensor_id"),
                    "power_series": prediction.get("power_series", []),
                    "baseload_series": prediction.get("baseload_series", []),
                    "state_series": prediction.get("state_series", []),
                    "state_summary": prediction.get("state_summary", {}),
                })
            result_payload = {"predictions": predictions}
        else:
            entry = model_entries[0]
            prediction = spool.load_model_payload(entry["safe_name"]) if spool is not None else {
                "power_series": [],
                "baseload_series": [],
                "state_series": [],
                "state_summary": {},
            }
            result_payload = {"result": prediction}
        emit({"done": True, **(result_payload or {})})
        return 0
    finally:
        if spool is not None:
            spool.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
