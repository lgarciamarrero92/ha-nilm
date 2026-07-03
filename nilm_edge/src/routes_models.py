import os
import time
import uuid
import asyncio
import gc
import json
import sys
import tempfile
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
from aiohttp import web

import app_state
from embedding_store import bundle_models_dir, delete_embedding_files, list_saved_models, load_embedding_metadata, save_embedding_metadata
from entity_slug import slugify_entity_suffix
from ha_client import HistoryQuery, fetch_history_points
from model_registry import get_bundle_by_id, make_model_key, parse_model_key
from prepare_training_data import (
    QueryExtractor,
    build_uniform_grid,
    load_model_settings,
    parse_mains_points,
    zoh_resample_to_grid,
)
from refquery import RefQueryDisaggregator

PREVIEW_HISTORY_TIMEOUT_S = 30.0
PREVIEW_HISTORY_CHUNK_HOURS = 1.0
PREVIEW_JOB_TTL_S = 15 * 60


def _batch_size():
    return app_state.get_batch_size()


def _realtime_entity_ids(appliance_name: str, bundle_id: str) -> dict:
    suffix_source = f"{appliance_name}_{bundle_id}" if bundle_id else appliance_name
    suffix = slugify_entity_suffix(suffix_source)
    return {
        "power": f"sensor.nilm_{suffix}_power",
        "energy": f"sensor.nilm_{suffix}_energy_consumed",
        "state": f"binary_sensor.nilm_{suffix}_on",
    }


def _cleanup_preview_result_file(path):
    result_path = str(path or "").strip()
    if not result_path:
        return
    try:
        if os.path.exists(result_path):
            os.remove(result_path)
    except OSError:
        pass


def _persist_preview_result(result):
    fd, result_path = tempfile.mkstemp(prefix="nilm_preview_cache_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {}, f, ensure_ascii=False)
    except Exception:
        _cleanup_preview_result_file(result_path)
        raise
    return result_path


def _load_preview_result(path):
    result_path = str(path or "").strip()
    if not result_path or not os.path.exists(result_path):
        return {}
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _adopt_preview_result_file(path):
    source_path = str(path or "").strip()
    if not source_path or not os.path.exists(source_path):
        return ""
    fd, adopted_path = tempfile.mkstemp(prefix="nilm_preview_cache_", suffix=".json")
    os.close(fd)
    try:
        os.replace(source_path, adopted_path)
    except OSError:
        shutil.copy2(source_path, adopted_path)
        _cleanup_preview_result_file(source_path)
    return adopted_path


async def _append_preview_job_chunk(app, job_id, chunk_path, chunk_index):
    async with app["preview_jobs_lock"]:
        current = app["preview_jobs"].get(job_id, {})
        queue = list(current.get("chunk_queue") or [])
        queue.append({
            "path": str(chunk_path or "").strip(),
            "index": int(chunk_index or (len(queue) + 1)),
        })
        current["chunk_queue"] = queue
        app["preview_jobs"][job_id] = current


async def _pop_preview_job_chunk(app, job_id):
    async with app["preview_jobs_lock"]:
        current = app["preview_jobs"].get(job_id)
        if not current:
            return None
        queue = list(current.get("chunk_queue") or [])
        if not queue:
            return None
        chunk = dict(queue.pop(0))
        current["chunk_queue"] = queue
        app["preview_jobs"][job_id] = current
        return chunk


async def _purge_stale_preview_jobs(app):
    now_ts = time.time()
    async with app["preview_jobs_lock"]:
        stale_job_ids = []
        stale_result_paths = []
        for job_id, job in list(app["preview_jobs"].items()):
            updated_at_raw = str(job.get("updated_at") or "").strip()
            try:
                updated_at_ts = datetime.fromisoformat(updated_at_raw).timestamp() if updated_at_raw else None
            except ValueError:
                updated_at_ts = None
            if updated_at_ts is None or (now_ts - updated_at_ts) <= PREVIEW_JOB_TTL_S:
                continue
            stale_job_ids.append(job_id)
            stale_result_paths.append(job.get("result_path"))
            for chunk in list(job.get("chunk_queue") or []):
                stale_result_paths.append(chunk.get("path"))

        for job_id in stale_job_ids:
            app["preview_jobs"].pop(job_id, None)

    for result_path in stale_result_paths:
        _cleanup_preview_result_file(result_path)


async def _set_preview_job(app, job_id, patch):
    async with app["preview_jobs_lock"]:
        current = app["preview_jobs"].get(job_id, {})
        if "result_path" in patch and patch.get("result_path") != current.get("result_path"):
            old_result_path = current.get("result_path")
            if old_result_path:
                _cleanup_preview_result_file(old_result_path)
        current.update(patch)
        app["preview_jobs"][job_id] = current


async def _get_preview_job(app, job_id):
    async with app["preview_jobs_lock"]:
        return dict(app["preview_jobs"].get(job_id) or {})


async def _pop_preview_job(app, job_id):
    async with app["preview_jobs_lock"]:
        return dict(app["preview_jobs"].pop(job_id, {}) or {})


async def _stream_preview_worker(payload):
    fd, input_path = tempfile.mkstemp(prefix="nilm_preview_", suffix=".json")
    os.close(fd)
    fd_out, output_path = tempfile.mkstemp(prefix="nilm_preview_result_", suffix=".json")
    os.close(fd_out)
    worker_path = os.path.join(os.path.dirname(__file__), "edge_preview_worker.py")
    try:
        payload = dict(payload or {})
        payload["result_path"] = output_path
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            worker_path,
            input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            raw = line.decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                update = json.loads(raw)
            except Exception:
                continue
            if isinstance(update, dict):
                if update.get("done") and update.get("result_path"):
                    update = dict(update)
                    update["result_path"] = _adopt_preview_result_file(update.get("result_path"))
                if update.get("chunk_path"):
                    update = dict(update)
                    update["chunk_path"] = _adopt_preview_result_file(update.get("chunk_path"))
                yield update

        stderr_bytes = await proc.stderr.read() if proc.stderr is not None else b""
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"preview worker failed with code {rc}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


async def _fetch_preview_history_points(fetch_start_dt, end_dt):
    entity_id = (app_state.current_config.get("main_sensor_id") or "").strip()
    if not entity_id:
        raise ValueError("No mains sensor configured. Please select and save a mains sensor first.")
    chunk_delta = timedelta(hours=PREVIEW_HISTORY_CHUNK_HOURS)
    chunk_bounds = []
    cursor = fetch_start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + chunk_delta, end_dt)
        chunk_bounds.append((cursor, chunk_end))
        cursor = chunk_end

    if not chunk_bounds:
        chunk_bounds.append((fetch_start_dt, end_dt))

    all_points = []
    seen = set()
    total_chunks = len(chunk_bounds)

    for index, (chunk_start, chunk_end) in enumerate(chunk_bounds, start=1):
        chunk_points = await asyncio.wait_for(
            fetch_history_points(
                app_state.HA_REST_API_URL,
                app_state.TOKEN,
                HistoryQuery(
                    entity_id=entity_id,
                    start_dt=chunk_start,
                    end_dt=chunk_end,
                    minimal_response=True,
                    max_span_days=7,
                ),
            ),
            timeout=PREVIEW_HISTORY_TIMEOUT_S,
        )

        for ts, value in chunk_points:
            key = (float(ts), float(value))
            if key in seen:
                continue
            seen.add(key)
            all_points.append((float(ts), float(value)))

        yield {
            "phase": "history",
            "processed": index,
            "total": total_chunks,
        }

    all_points.sort(key=lambda item: item[0])
    yield {
        "phase": "history_ready",
        "processed": len(all_points),
        "total": len(all_points),
        "points": all_points,
    }


def _normalize_preview_points(raw_points):
    normalized = []
    seen = set()
    for point in raw_points or []:
        if not isinstance(point, dict):
            continue
        raw_x = point.get("x")
        raw_y = point.get("y")
        try:
            ts = float(raw_x)
            if ts > 1e12:
                ts /= 1000.0
            value = float(raw_y)
        except (TypeError, ValueError):
            continue
        key = (ts, value)
        if key in seen:
            continue
        seen.add(key)
        normalized.append((ts, value))
    normalized.sort(key=lambda item: item[0])
    return normalized


async def _fetch_preview_history_points_range(fetch_start_dt, end_dt):
    if fetch_start_dt >= end_dt:
        return []

    points = []
    async for history_update in _fetch_preview_history_points(fetch_start_dt, end_dt):
        if history_update.get("phase") == "history_ready":
            points = history_update.get("points") or []
    return points


def _filter_points_to_range(points, start_ts, end_ts):
    return [
        (float(ts), float(value))
        for ts, value in (points or [])
        if float(ts) >= start_ts and float(ts) <= end_ts
    ]


async def _load_preview_points(start_dt, end_dt, provided_points=None):
    points = _filter_points_to_range(
        _normalize_preview_points(provided_points),
        start_dt.timestamp(),
        end_dt.timestamp(),
    )
    if points:
        yield {
            "processed": len(points),
            "total": len(points),
            "phase": "history_ready",
            "points": points,
        }
        return

    try:
        async for history_update in _fetch_preview_history_points(start_dt, end_dt):
            if history_update.get("phase") == "history_ready":
                history_update = dict(history_update)
                history_update["points"] = _filter_points_to_range(
                    history_update.get("points") or [],
                    start_dt.timestamp(),
                    end_dt.timestamp(),
                )
            yield history_update
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Loading mains history took too long ({int(PREVIEW_HISTORY_TIMEOUT_S)}s). "
            "Please reduce the selected range or check the Home Assistant connection."
        ) from exc


def _build_offline_preview_inputs(points, *, inference_dir: str, num_threads: int = 2, align_grid: str = "start", max_hold_factor: float = 5.0):
    parsed_points = parse_mains_points([{"x": float(ts) * 1000.0, "y": float(value)} for ts, value in (points or [])])
    if len(parsed_points) < 2:
        return {
            "windows": np.zeros((0, 0), dtype=np.float32),
            "label_times_ms": np.zeros((0,), dtype=np.int64),
            "mains_at_label": np.zeros((0,), dtype=np.float32),
            "baseload_at_label": np.zeros((0,), dtype=np.float32),
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
            "windows": np.zeros((0, 0), dtype=np.float32),
            "label_times_ms": np.zeros((0,), dtype=np.int64),
            "mains_at_label": np.zeros((0,), dtype=np.float32),
            "baseload_at_label": np.zeros((0,), dtype=np.float32),
            "inference_dir": inference_dir,
            "num_threads": int(num_threads),
        }

    max_hold_s = float(max_hold_factor) * dt if max_hold_factor and max_hold_factor > 0 else None
    fill_value_w = float(settings.query_mean)
    y_grid, mains_valid_mask = zoh_resample_to_grid(
        parsed_points,
        grid_t,
        max_hold_s=max_hold_s,
        fill_value=fill_value_w,
        return_valid_mask=True,
    )

    M = int(grid_t.size)
    N = M - (T - 1)
    windows_raw = np.lib.stride_tricks.sliding_window_view(y_grid.astype(np.float32), window_shape=T)
    baseload_per_window = np.min(windows_raw, axis=1).astype(np.float32)
    windows_shifted = windows_raw - baseload_per_window[:, None]
    query_std = settings.query_std if abs(settings.query_std) > 1e-12 else 1.0
    windows = ((windows_shifted - float(settings.query_mean)) / float(query_std)).astype(np.float32)
    valid_windows_mask = np.all(
        np.lib.stride_tricks.sliding_window_view(mains_valid_mask.astype(np.uint8), window_shape=T) == 1,
        axis=1,
    )

    grid_t_end = grid_t[(T - 1):]
    offset = (T - 1 - pred_idx) * dt
    grid_t_label = grid_t_end - offset
    mains_at_label = y_grid[pred_idx:(pred_idx + N)].astype(np.float32)

    windows = windows[valid_windows_mask]
    grid_t_label = grid_t_label[valid_windows_mask]
    mains_at_label = mains_at_label[valid_windows_mask]
    baseload_per_window = baseload_per_window[valid_windows_mask]

    if windows.shape[0] == 0:
        return {
            "windows": np.zeros((0, 0), dtype=np.float32),
            "label_times_ms": np.zeros((0,), dtype=np.int64),
            "mains_at_label": np.zeros((0,), dtype=np.float32),
            "baseload_at_label": np.zeros((0,), dtype=np.float32),
            "inference_dir": inference_dir,
            "num_threads": int(num_threads),
        }

    return {
        "windows": windows,
        "label_times_ms": np.round(grid_t_label * 1000.0).astype(np.int64),
        "mains_at_label": mains_at_label.astype(np.float32),
        "baseload_at_label": baseload_per_window.astype(np.float32),
        "inference_dir": inference_dir,
        "num_threads": int(num_threads),
    }


def _score_preview_embeddings(preview_disaggregator, safe_names, preview_inputs):
    embeddings = np.asarray(preview_inputs.get("embeddings"), dtype=np.float32)
    label_times_ms = np.asarray(preview_inputs.get("label_times_ms"), dtype=np.int64)
    mains_at_label = np.asarray(preview_inputs.get("mains_at_label"), dtype=np.float32)
    baseload_at_label = np.asarray(preview_inputs.get("baseload_at_label"), dtype=np.float32)

    prediction_map = {
        safe_name: {
            "power_series": [],
            "baseload_series": [],
            "state_series": [],
        }
        for safe_name in safe_names
    }

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return prediction_map

    model_state = {}
    for safe_name in safe_names:
        ref = preview_disaggregator._load_embedding(safe_name)
        params = preview_disaggregator._load_appliance_params(safe_name)
        model_state[safe_name] = (ref, float(params.get("onoff_threshold", 0.5)))

    for idx in range(embeddings.shape[0]):
        query_emb = np.asarray(embeddings[idx], dtype=np.float32).reshape(1, -1)
        target_ms = int(label_times_ms[idx])
        baseload_value = float(max(0.0, baseload_at_label[idx])) if idx < baseload_at_label.size else 0.0
        available_appliance_power = float(max(0.0, float(mains_at_label[idx]) - baseload_value)) if idx < mains_at_label.size else 0.0

        for safe_name in safe_names:
            ref, onoff_threshold = model_state[safe_name]
            power_w, onoff_value, _power_norm = preview_disaggregator._run_head(ref, query_emb)
            power_w = float(max(0.0, power_w))
            power_w = float(min(power_w, available_appliance_power))
            if onoff_value < onoff_threshold:
                power_w = 0.0

            prediction_map[safe_name]["power_series"].append({"x": target_ms, "y": power_w})
            prediction_map[safe_name]["baseload_series"].append({"x": target_ms, "y": baseload_value})
            prediction_map[safe_name]["state_series"].append({
                "x": target_ms,
                "y": int(onoff_value >= onoff_threshold),
                "score": float(onoff_value),
                "threshold": float(onoff_threshold),
            })

    return prediction_map


def _summarize_state_series(state_series):
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


async def _extract_preview_embeddings_with_progress(preview_inputs):
    windows = np.asarray(preview_inputs.get("windows"), dtype=np.float32)
    inference_dir = str(preview_inputs.get("inference_dir") or "")
    num_threads = int(preview_inputs.get("num_threads") or 2)
    label_times_ms = np.asarray(preview_inputs.get("label_times_ms"), dtype=np.int64)
    mains_at_label = np.asarray(preview_inputs.get("mains_at_label"), dtype=np.float32)
    baseload_at_label = np.asarray(preview_inputs.get("baseload_at_label"), dtype=np.float32)

    if not inference_dir or windows.ndim != 2 or windows.shape[0] == 0:
        yield {
            "phase": "embeddings",
            "processed": 0,
            "total": 0,
            "preview_inputs": {
                "embeddings": np.zeros((0, 0), dtype=np.float32),
                "label_times_ms": np.zeros((0,), dtype=np.int64),
                "mains_at_label": np.zeros((0,), dtype=np.float32),
                "baseload_at_label": np.zeros((0,), dtype=np.float32),
            },
        }
        return

    extractor = QueryExtractor(os.path.join(inference_dir, "extractor.tflite"), num_threads=num_threads)
    X = None
    embs = None
    ok = None
    try:
        X = extractor.build_input_batch(windows)
        out_shape = tuple(extractor.out[0]["shape"])
        D = int(np.prod(out_shape)) if len(out_shape) >= 1 else 0
        if D <= 0:
            extractor.interp.set_tensor(extractor.in_index, X[0].astype(extractor.in_dtype, copy=False))
            extractor.interp.invoke()
            D = int(np.asarray(extractor.interp.get_tensor(extractor.out_index)).size)

        N = int(X.shape[0])
        embs = np.zeros((N, D), dtype=np.float32)
        ok = np.zeros((N,), dtype=bool)
        last_emit = 0.0

        yield {"phase": "embeddings", "processed": 0, "total": N}

        for i in range(N):
            try:
                xi = X[i]
                if np.all(np.isfinite(xi)):
                    extractor.interp.set_tensor(extractor.in_index, xi.astype(extractor.in_dtype, copy=False))
                    extractor.interp.invoke()
                    emb = np.asarray(extractor.interp.get_tensor(extractor.out_index), dtype=np.float32).reshape(-1)
                    if np.linalg.norm(emb) < 1e-9:
                        emb = emb + extractor.eps
                    embs[i, :] = emb
                    ok[i] = True
            except Exception:
                pass

            now_mono = time.monotonic()
            if i == 0 or i == N - 1 or now_mono - last_emit >= 0.05:
                yield {"phase": "embeddings", "processed": i + 1, "total": N}
                last_emit = now_mono
            if (i + 1) % 32 == 0:
                await asyncio.sleep(0)

        yield {
            "phase": "embeddings_ready",
            "processed": int(np.sum(ok)),
            "total": N,
            "preview_inputs": {
                "embeddings": embs[ok],
                "label_times_ms": label_times_ms[ok],
                "mains_at_label": mains_at_label[ok],
                "baseload_at_label": baseload_at_label[ok],
            },
        }
    finally:
        del X
        del embs
        del ok
        del extractor
        gc.collect()


async def _score_preview_embeddings_with_progress(preview_disaggregator, safe_names, preview_inputs):
    embeddings = np.asarray(preview_inputs.get("embeddings"), dtype=np.float32)
    label_times_ms = np.asarray(preview_inputs.get("label_times_ms"), dtype=np.int64)
    mains_at_label = np.asarray(preview_inputs.get("mains_at_label"), dtype=np.float32)
    baseload_at_label = np.asarray(preview_inputs.get("baseload_at_label"), dtype=np.float32)

    prediction_map = {
        safe_name: {
            "power_series": [],
            "baseload_series": [],
            "state_series": [],
        }
        for safe_name in safe_names
    }

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        yield {
            "phase": "inference",
            "processed": 0,
            "total": 0,
            "prediction_map": prediction_map,
        }
        return

    model_state = {}
    try:
        for safe_name in safe_names:
            ref = preview_disaggregator._load_embedding(safe_name)
            params = preview_disaggregator._load_appliance_params(safe_name)
            model_state[safe_name] = (ref, float(params.get("onoff_threshold", 0.5)))

        N = int(embeddings.shape[0])
        last_emit = 0.0
        yield {"phase": "inference", "processed": 0, "total": N}

        for idx in range(N):
            query_emb = np.asarray(embeddings[idx], dtype=np.float32).reshape(1, -1)
            target_ms = int(label_times_ms[idx])
            baseload_value = float(max(0.0, baseload_at_label[idx])) if idx < baseload_at_label.size else 0.0
            available_appliance_power = float(max(0.0, float(mains_at_label[idx]) - baseload_value)) if idx < mains_at_label.size else 0.0

            for safe_name in safe_names:
                ref, onoff_threshold = model_state[safe_name]
                power_w, onoff_value, _power_norm = preview_disaggregator._run_head(ref, query_emb)
                power_w = float(max(0.0, power_w))
                power_w = float(min(power_w, available_appliance_power))
                if onoff_value < onoff_threshold:
                    power_w = 0.0

                prediction_map[safe_name]["power_series"].append({"x": target_ms, "y": power_w})
                prediction_map[safe_name]["baseload_series"].append({"x": target_ms, "y": baseload_value})
                prediction_map[safe_name]["state_series"].append({
                    "x": target_ms,
                    "y": int(onoff_value >= onoff_threshold),
                    "score": float(onoff_value),
                    "threshold": float(onoff_threshold),
                })

            now_mono = time.monotonic()
            if idx == 0 or idx == N - 1 or now_mono - last_emit >= 0.05:
                yield {"phase": "inference", "processed": idx + 1, "total": N}
                last_emit = now_mono
            if (idx + 1) % 32 == 0:
                await asyncio.sleep(0)

        yield {
            "phase": "inference_done",
            "processed": N,
            "total": N,
            "prediction_map": prediction_map,
        }
    finally:
        del model_state
        gc.collect()


async def _build_preview_result(bundle_id, safe_name, start_dt, end_dt, provided_points=None):
    bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
    if bundle is None:
        raise ValueError(f"Unknown bundle_id: {bundle_id}")

    preview_disaggregator = None
    points = []
    preview_inputs = None
    extracted_preview_inputs = None
    prediction_map = None
    try:
        preview_disaggregator = RefQueryDisaggregator(
            inference_dir=bundle.inference_dir,
            embeddings_dir=bundle_models_dir(app_state.MODELS_ROOT, bundle_id),
            num_threads=2,
            history_fetcher=None,
            top_k=None,
        )

        async for history_update in _load_preview_points(start_dt, end_dt, provided_points=provided_points):
            if history_update.get("phase") == "history_ready":
                points = history_update.get("points") or []
                yield {
                    "processed": int(history_update.get("processed", 0)),
                    "total": int(history_update.get("total", 0)),
                    "phase": "history_ready",
                }
            else:
                yield history_update

        if not points:
            yield {
                "done": True,
                "power_series": [],
                "state_series": [],
                "processed": 0,
                "total": 0,
            }
            return

        preview_inputs = _build_offline_preview_inputs(points, inference_dir=bundle.inference_dir, num_threads=2, align_grid="start", max_hold_factor=5.0)
        async for update in _extract_preview_embeddings_with_progress(preview_inputs):
            if update.get("phase") == "embeddings_ready":
                extracted_preview_inputs = update.get("preview_inputs") or {}
            else:
                yield {
                    "processed": int(update.get("processed", 0)),
                    "total": int(update.get("total", 0)),
                    "phase": "embeddings",
                }
        preview_inputs = extracted_preview_inputs or {}
        embeddings = np.asarray(preview_inputs.get("embeddings"), dtype=np.float32)
        total = int(embeddings.shape[0])
        if total <= 0:
            yield {
                "done": True,
                "power_series": [],
                "baseload_series": [],
                "state_series": [],
                "processed": 0,
                "total": 0,
            }
            return

        async for update in _score_preview_embeddings_with_progress(preview_disaggregator, [safe_name], preview_inputs):
            if update.get("phase") == "inference_done":
                prediction_map = update.get("prediction_map") or {}
            else:
                yield {
                    "processed": int(update.get("processed", 0)),
                    "total": int(update.get("total", 0)),
                    "phase": "inference",
                }
        prediction_map = prediction_map or {}
        prediction = prediction_map.get(safe_name) or {}
        state_summary = _summarize_state_series(prediction.get("state_series", []))
        print(
            f"Preview summary appliance={safe_name} "
            f"n_points={state_summary.get('n_points')} "
            f"max_score={state_summary.get('max_score')} "
            f"mean_score={state_summary.get('mean_score')} "
            f"threshold={state_summary.get('threshold')} "
            f"n_above_threshold={state_summary.get('n_above_threshold')}",
            flush=True,
        )

        yield {
            "done": True,
            "power_series": prediction.get("power_series", []),
            "baseload_series": prediction.get("baseload_series", []),
            "state_series": prediction.get("state_series", []),
            "state_summary": state_summary,
            "processed": total,
            "total": total,
        }
    finally:
        del points
        del preview_inputs
        del extracted_preview_inputs
        del prediction_map
        del preview_disaggregator
        gc.collect()


async def _build_preview_all_results(model_entries, start_dt, end_dt, provided_points=None):
    if not model_entries:
        yield {"done": True, "predictions": [], "processed": 0, "total": 0}
        return

    grouped_models = {}
    for model_entry in model_entries:
        bundle_id = str(model_entry.get("bundle_id") or "").strip()
        safe_name = str(model_entry.get("appliance_name") or "").strip()
        if not bundle_id or not safe_name:
            continue
        grouped_models.setdefault(bundle_id, []).append({
            "model_key": make_model_key(bundle_id, safe_name),
            "model_name": safe_name,
            "safe_name": safe_name,
        })

    if not grouped_models:
        yield {"done": True, "predictions": [], "processed": 0, "total": 0}
        return

    all_predictions = []

    try:
        for bundle_id, bundle_models in grouped_models.items():
            bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
            if bundle is None:
                continue

            preview_disaggregator = None
            points = []
            preview_inputs = None
            extracted_preview_inputs = None
            scored_predictions = None
            try:
                preview_disaggregator = RefQueryDisaggregator(
                    inference_dir=bundle.inference_dir,
                    embeddings_dir=bundle_models_dir(app_state.MODELS_ROOT, bundle_id),
                    num_threads=2,
                    history_fetcher=None,
                    top_k=None,
                )

                async for history_update in _load_preview_points(start_dt, end_dt, provided_points=provided_points):
                    if history_update.get("phase") == "history_ready":
                        points = history_update.get("points") or []
                        yield {
                            "processed": int(history_update.get("processed", 0)),
                            "total": int(history_update.get("total", 0)),
                            "phase": "history_ready",
                        }
                    else:
                        yield history_update

                if not points:
                    continue

                model_names = [item["safe_name"] for item in bundle_models]
                preview_inputs = _build_offline_preview_inputs(points, inference_dir=bundle.inference_dir, num_threads=2, align_grid="start", max_hold_factor=5.0)
                async for update in _extract_preview_embeddings_with_progress(preview_inputs):
                    if update.get("phase") == "embeddings_ready":
                        extracted_preview_inputs = update.get("preview_inputs") or {}
                    else:
                        yield {
                            "processed": int(update.get("processed", 0)),
                            "total": int(update.get("total", 0)),
                            "phase": "embeddings",
                        }
                preview_inputs = extracted_preview_inputs or {}
                embeddings = np.asarray(preview_inputs.get("embeddings"), dtype=np.float32)
                total = int(embeddings.shape[0])
                if total <= 0:
                    continue

                async for update in _score_preview_embeddings_with_progress(preview_disaggregator, model_names, preview_inputs):
                    if update.get("phase") == "inference_done":
                        scored_predictions = update.get("prediction_map") or {}
                    else:
                        yield {
                            "processed": int(update.get("processed", 0)),
                            "total": int(update.get("total", 0)),
                            "phase": "inference",
                        }
                scored_predictions = scored_predictions or {}

                for item in bundle_models:
                    prediction = scored_predictions.get(item["safe_name"]) or {}
                    power_series = prediction.get("power_series", [])
                    state_series = prediction.get("state_series", [])
                    state_summary = _summarize_state_series(state_series)

                    all_predictions.append({
                        "model_key": item["model_key"],
                        "model_name": item["model_name"],
                        "power_series": power_series,
                        "baseload_series": prediction.get("baseload_series", []),
                        "state_series": state_series,
                        "state_summary": state_summary,
                    })
            finally:
                del points
                del preview_inputs
                del extracted_preview_inputs
                del scored_predictions
                del preview_disaggregator
                gc.collect()

        yield {
            "done": True,
            "predictions": all_predictions,
            "processed": len(all_predictions),
            "total": len(all_predictions),
        }
    finally:
        del all_predictions
        gc.collect()


async def _run_preview_job(app, job_id, bundle_id, safe_name, start_dt, end_dt, provided_points=None):
    try:
        bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
        if bundle is None:
            raise ValueError(f"Unknown bundle_id: {bundle_id}")

        await _set_preview_job(app, job_id, {
            "status": "running",
            "phase": "history",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": 0,
            "total": 0,
            "percent": 5,
        })

        points = []
        async for history_update in _load_preview_points(start_dt, end_dt, provided_points=provided_points):
            if history_update.get("phase") == "history_ready":
                points = history_update.get("points") or []
                processed = int(history_update.get("processed", 0))
                total = int(history_update.get("total", 0))
                phase = "history_ready"
                percent = 25
            else:
                processed = int(history_update.get("processed", 0))
                total = int(history_update.get("total", 0))
                phase = "history"
                percent = max(5, min(24, int(round((processed / total) * 24)))) if total > 0 else 5

            await _set_preview_job(app, job_id, {
                "status": "running",
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": processed,
                "total": total,
                "percent": percent,
            })
            await asyncio.sleep(0)

        if not points:
            await _set_preview_job(app, job_id, {
                "status": "done",
                "phase": "done",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": 0,
                "total": 0,
                "percent": 100,
                "result": {
                    "power_series": [],
                    "baseload_series": [],
                    "state_series": [],
                    "state_summary": {},
                },
            })
            return

        payload = {
            "mode": "single",
            "stream_chunks": True,
            "batch_size": _batch_size(),
            "points": points,
            "models": [{
                "bundle_id": bundle_id,
                "safe_name": safe_name,
                "model_name": safe_name,
                "model_key": make_model_key(bundle_id, safe_name),
                "inference_dir": bundle.inference_dir,
                "embeddings_dir": bundle_models_dir(app_state.MODELS_ROOT, bundle_id),
            }],
        }

        async for update in _stream_preview_worker(payload):
            if update.get("chunk_path"):
                await _append_preview_job_chunk(
                    app,
                    job_id,
                    update.get("chunk_path"),
                    update.get("chunk_index"),
                )

            if update.get("done"):
                result_path = str(update.get("result_path") or "").strip()
                result_meta = {}
                if result_path:
                    loaded = _load_preview_result(result_path)
                    result_meta = dict(loaded.get("result") or {})
                state_summary = result_meta.get("state_summary") or {}
                print(
                    f"Preview summary appliance={safe_name} "
                    f"n_points={state_summary.get('n_points')} "
                    f"max_score={state_summary.get('max_score')} "
                    f"mean_score={state_summary.get('mean_score')} "
                    f"threshold={state_summary.get('threshold')} "
                    f"n_above_threshold={state_summary.get('n_above_threshold')}",
                    flush=True,
                )
                await _set_preview_job(app, job_id, {
                    "status": "done",
                    "phase": "done",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "processed": int(update.get("n_points") or state_summary.get("n_points") or 0),
                    "total": int(update.get("n_points") or state_summary.get("n_points") or 0),
                    "percent": 100,
                    "result": None,
                    "result_path": result_path,
                })
                return

            processed = int(update.get("processed", 0))
            total = int(update.get("total", 0))
            phase = str(update.get("phase") or "inference")
            explicit_percent = update.get("percent")
            if isinstance(explicit_percent, (int, float)):
                percent = max(0, min(100, int(round(float(explicit_percent)))))
            elif total > 0:
                if phase == "embeddings":
                    percent = max(26, min(55, 25 + int(round((processed / total) * 30))))
                elif phase == "inference":
                    percent = max(56, min(92, 55 + int(round((processed / total) * 37))))
                else:
                    percent = max(26, min(92, 25 + int(round((processed / total) * 67))))
            else:
                percent = 30

            await _set_preview_job(app, job_id, {
                "status": "running",
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": processed,
                "total": total,
                "percent": percent,
            })
            await asyncio.sleep(0)
    except Exception as exc:
        await _set_preview_job(app, job_id, {
            "status": "error",
            "phase": "error",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "message": str(exc),
        })
    finally:
        gc.collect()


async def _run_preview_all_job(app, job_id, model_entries, start_dt, end_dt, provided_points=None):
    try:
        await _set_preview_job(app, job_id, {
            "status": "running",
            "phase": "history",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": 0,
            "total": 0,
            "percent": 5,
        })

        points = []
        async for history_update in _load_preview_points(start_dt, end_dt, provided_points=provided_points):
            if history_update.get("phase") == "history_ready":
                points = history_update.get("points") or []
                processed = int(history_update.get("processed", 0))
                total = int(history_update.get("total", 0))
                phase = "history_ready"
                percent = 25
            else:
                processed = int(history_update.get("processed", 0))
                total = int(history_update.get("total", 0))
                phase = "history"
                percent = max(5, min(24, int(round((processed / total) * 24)))) if total > 0 else 5
            await _set_preview_job(app, job_id, {
                "status": "running",
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": processed,
                "total": total,
                "percent": percent,
            })
            await asyncio.sleep(0)

        if not points:
            result_path = _persist_preview_result({
                "predictions": [],
            })
            await _set_preview_job(app, job_id, {
                "status": "done",
                "phase": "done",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": 0,
                "total": 0,
                "percent": 100,
                "result": None,
                "result_path": result_path,
            })
            return

        payload_models = []
        for model_entry in model_entries:
            bundle_id = str(model_entry.get("bundle_id") or "").strip()
            safe_name = str(model_entry.get("appliance_name") or "").strip()
            bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
            if bundle is None or not safe_name:
                continue
            payload_models.append({
                "bundle_id": bundle_id,
                "safe_name": safe_name,
                "model_name": safe_name,
                "model_key": make_model_key(bundle_id, safe_name),
                "inference_dir": bundle.inference_dir,
                "embeddings_dir": bundle_models_dir(app_state.MODELS_ROOT, bundle_id),
            })

        async for update in _stream_preview_worker({
            "mode": "all",
            "batch_size": _batch_size(),
            "points": points,
            "models": payload_models,
        }):
            if update.get("chunk_path"):
                await _append_preview_job_chunk(
                    app,
                    job_id,
                    update.get("chunk_path"),
                    update.get("chunk_index"),
                )

            if update.get("done"):
                prediction_count = int(update.get("prediction_count") or 0)
                await _set_preview_job(app, job_id, {
                    "status": "done",
                    "phase": "done",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "processed": prediction_count,
                    "total": prediction_count,
                    "percent": 100,
                    "result": None,
                    "result_path": None,
                })
                return

            processed = int(update.get("processed", 0))
            total = int(update.get("total", 0))
            phase = str(update.get("phase") or "inference")
            explicit_percent = update.get("percent")
            if isinstance(explicit_percent, (int, float)):
                percent = max(0, min(100, int(round(float(explicit_percent)))))
            elif total > 0:
                if phase == "embeddings":
                    percent = max(26, min(55, 25 + int(round((processed / total) * 30))))
                elif phase == "inference":
                    percent = max(56, min(92, 55 + int(round((processed / total) * 37))))
                else:
                    percent = max(26, min(92, 25 + int(round((processed / total) * 67))))
            else:
                percent = 30
            await _set_preview_job(app, job_id, {
                "status": "running",
                "phase": phase,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed": processed,
                "total": total,
                "percent": percent,
            })
            await asyncio.sleep(0)
    except Exception as exc:
        await _set_preview_job(app, job_id, {
            "status": "error",
            "phase": "error",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "message": str(exc),
        })
    finally:
        gc.collect()


async def get_embeddings_handler(request):
    try:
        embeddings = []
        for model_entry in list_saved_models(app_state.MODELS_ROOT):
            bundle_id = model_entry["bundle_id"]
            name = model_entry["appliance_name"]
            path = model_entry["embedding_path"]
            bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
            if bundle is None:
                continue
            stat = os.stat(path)
            metadata = load_embedding_metadata(bundle_models_dir(app_state.MODELS_ROOT, bundle_id), name) or {}
            embeddings.append({
                "model_key": make_model_key(bundle_id, name),
                "name": name,
                "bundle_id": bundle_id,
                "bundle_mode": bundle.mode,
                "bundle_version": bundle.model_version,
                "bundle_label": bundle.label,
                "file_name": os.path.basename(path),
                "path": path,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "source": "trained",
                "metadata": metadata,
                "realtime_entity_ids": _realtime_entity_ids(name, bundle_id),
                "deletable": True,
            })

        return web.json_response({"status": "success", "embeddings": embeddings})
    except Exception as exc:
        print(f"Error handling GET /embeddings: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def get_model_bundles_handler(request):
    try:
        bundles = [
            {
                "bundle_id": bundle.bundle_id,
                "mode": bundle.mode,
                "version": bundle.model_version,
                "label": bundle.label,
                "is_default_for_training": bundle.is_default_for_training,
            }
            for bundle in app_state.model_bundles
        ]
        return web.json_response({"status": "success", "bundles": bundles})
    except Exception as exc:
        print(f"Error handling GET /model-bundles: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def delete_embedding_handler(request):
    try:
        model_key = str(request.match_info.get("name") or "").strip()
        if not model_key:
            return web.json_response({"status": "error", "message": "Missing embedding name"}, status=400)

        bundle_id, safe_name = parse_model_key(model_key)
        bundle_dir = bundle_models_dir(app_state.MODELS_ROOT, bundle_id)
        deleted = delete_embedding_files(bundle_dir, safe_name)
        if not deleted["embedding"] and not deleted["metadata"]:
            return web.json_response({"status": "error", "message": "Embedding not found"}, status=404)

        app_state.reload_algorithm_config()
        return web.json_response({"status": "success", "message": f"Appliance model '{safe_name}' deleted"})
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        print(f"Error handling DELETE /embeddings/{{name}}: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def update_embedding_handler(request):
    try:
        model_key = str(request.match_info.get("name") or "").strip()
        if not model_key:
            return web.json_response({"status": "error", "message": "Missing embedding name"}, status=400)

        bundle_id, safe_name = parse_model_key(model_key)
        bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
        if bundle is None:
            return web.json_response({"status": "error", "message": f"Unknown bundle_id: {bundle_id}"}, status=404)

        if request.method == "POST":
            action = str(request.query.get("action") or "").strip().lower()
            if action != "update":
                return web.json_response({"status": "error", "message": "Missing/invalid action. Use ?action=update"}, status=400)

        data = await request.json()
        if not isinstance(data, dict):
            return web.json_response({"status": "error", "message": "Invalid JSON body"}, status=400)

        bundle_dir = bundle_models_dir(app_state.MODELS_ROOT, bundle_id)
        metadata = load_embedding_metadata(bundle_dir, safe_name) or {}

        if "publish_online" in data:
            if bundle.mode != "online":
                return web.json_response({"status": "error", "message": "Only online bundles can publish live to Home Assistant"}, status=400)
            metadata["publish_online"] = bool(data.get("publish_online"))

        save_embedding_metadata(bundle_dir, safe_name, metadata)
        app_state.reload_algorithm_config()
        return web.json_response({"status": "success", "metadata": metadata})
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        print(f"Error handling PATCH /embeddings/{{name}}: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def start_preview_embedding_handler(request):
    try:
        await _purge_stale_preview_jobs(request.app)
        model_key = str(request.match_info.get("name") or "").strip()
        if not model_key:
            return web.json_response({"status": "error", "message": "Missing embedding name"}, status=400)

        bundle_id, safe_name = parse_model_key(model_key)
        bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
        if bundle is None:
            return web.json_response({"status": "error", "message": f"Unknown bundle_id: {bundle_id}"}, status=404)

        if request.method == "POST":
            data = await request.json()
            if not isinstance(data, dict):
                return web.json_response({"status": "error", "message": "Invalid JSON body"}, status=400)
            start_raw = str(data.get("start") or "").strip()
            end_raw = str(data.get("end") or "").strip()
            provided_points = data.get("mains_points")
        else:
            start_raw = str(request.query.get("start") or "").strip()
            end_raw = str(request.query.get("end") or "").strip()
            provided_points = None
        if not start_raw or not end_raw:
            return web.json_response({"status": "error", "message": "Missing start/end query parameters"}, status=400)

        start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        if start_dt >= end_dt:
            return web.json_response({"status": "error", "message": "start must be before end"}, status=400)

        job_id = uuid.uuid4().hex
        await _set_preview_job(request.app, job_id, {
            "status": "queued",
            "phase": "queued",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": 0,
            "total": 0,
            "percent": 0,
            "message": None,
        })
        asyncio.create_task(_run_preview_job(request.app, job_id, bundle_id, safe_name, start_dt, end_dt, provided_points=provided_points))
        return web.json_response({"status": "accepted", "job_id": job_id})
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        print(f"Error handling POST /embeddings/{{name}}/preview: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def start_preview_all_embeddings_handler(request):
    try:
        await _purge_stale_preview_jobs(request.app)
        data = await request.json()
        if not isinstance(data, dict):
            return web.json_response({"status": "error", "message": "Invalid JSON body"}, status=400)

        start_raw = str(data.get("start") or "").strip()
        end_raw = str(data.get("end") or "").strip()
        provided_points = data.get("mains_points")
        if not start_raw or not end_raw:
            return web.json_response({"status": "error", "message": "Missing start/end query parameters"}, status=400)

        start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        if start_dt >= end_dt:
            return web.json_response({"status": "error", "message": "start must be before end"}, status=400)

        model_entries = []
        for model_entry in list_saved_models(app_state.MODELS_ROOT):
            bundle_id = model_entry["bundle_id"]
            safe_name = model_entry["appliance_name"]
            bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
            if bundle is None:
                continue
            model_entries.append({
                "bundle_id": bundle_id,
                "appliance_name": safe_name,
            })

        if not model_entries:
            return web.json_response({"status": "error", "message": "No appliance models found yet."}, status=400)

        job_id = uuid.uuid4().hex
        await _set_preview_job(request.app, job_id, {
            "status": "queued",
            "phase": "queued",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed": 0,
            "total": 0,
            "percent": 0,
            "message": None,
        })
        asyncio.create_task(_run_preview_all_job(request.app, job_id, model_entries, start_dt, end_dt, provided_points=provided_points))
        return web.json_response({"status": "accepted", "job_id": job_id})
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        print(f"Error handling POST /embeddings/preview-all: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def preview_embedding_status_handler(request):
    try:
        await _purge_stale_preview_jobs(request.app)
        job_id = str(request.match_info.get("job_id") or "").strip()
        if not job_id:
            return web.json_response({"status": "error", "message": "Missing job_id"}, status=400)

        job = await _get_preview_job(request.app, job_id)
        if not job:
            return web.json_response({"status": "error", "message": "Preview job not found"}, status=404)

        chunk_meta = await _pop_preview_job_chunk(request.app, job_id)
        chunk_payload = None
        if chunk_meta:
            chunk_path = str(chunk_meta.get("path") or "").strip()
            if chunk_path:
                chunk_payload = _load_preview_result(chunk_path)
                _cleanup_preview_result_file(chunk_path)
            job = await _get_preview_job(request.app, job_id)

        remaining_chunks = list(job.get("chunk_queue") or [])
        if job.get("status") == "done" and remaining_chunks:
            job["status"] = "running"
            job["phase"] = "inference"
            job["percent"] = min(99, int(job.get("percent") or 99))

        if job.get("status") in {"done", "error"}:
            remaining_chunks = list(job.get("chunk_queue") or [])
            if not remaining_chunks:
                job = await _pop_preview_job(request.app, job_id)
                result_path = job.get("result_path")
                if result_path:
                    loaded_result = _load_preview_result(result_path)
                    if isinstance(loaded_result, dict) and "result" in loaded_result:
                        job["result"] = loaded_result.get("result") or {}
                    else:
                        job["result"] = loaded_result
                    job.pop("result_path", None)
                    _cleanup_preview_result_file(result_path)
                gc.collect()

        job.pop("chunk_queue", None)
        if chunk_payload is not None:
            job["chunk"] = chunk_payload

        return web.json_response({
            "status": "success",
            "job_id": job_id,
            **job,
        })
    except Exception as exc:
        print(f"Error handling GET /preview-jobs/{{job_id}}: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


def register_model_routes(app, ingress_url_base):
    if "preview_jobs" not in app:
        app["preview_jobs"] = {}
    if "preview_jobs_lock" not in app:
        app["preview_jobs_lock"] = asyncio.Lock()

    app.router.add_get(ingress_url_base + "model-bundles", get_model_bundles_handler)
    app.router.add_get(ingress_url_base + "embeddings", get_embeddings_handler)
    app.router.add_post(ingress_url_base + "embeddings/{name}/preview", start_preview_embedding_handler)
    app.router.add_post(ingress_url_base + "embeddings/preview-all", start_preview_all_embeddings_handler)
    app.router.add_get(ingress_url_base + "preview-jobs/{job_id}", preview_embedding_status_handler)
    app.router.add_patch(ingress_url_base + "embeddings/{name}", update_embedding_handler)
    app.router.add_post(ingress_url_base + "embeddings/{name}", update_embedding_handler)
    app.router.add_delete(ingress_url_base + "embeddings/{name}", delete_embedding_handler)
