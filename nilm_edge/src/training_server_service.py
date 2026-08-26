from __future__ import annotations

import os
import json
import uuid
import asyncio
import traceback
import gc
import ctypes
import sys
import gzip
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import inspect

import app_state
from training_server_client import (
    TrainingServerError,
    complete_training_upload_session,
    create_training_upload_session,
    probe_training_server_connection,
    poll_training_result,
    fetch_training_status,
    start_training_job_from_gzip_file,
    upload_training_chunk,
)
from embedding_store import bundle_models_dir, load_embedding_metadata
from training_payload import training_server_payload_from_prepared, summarize_training_server_payload
from embedding_store import save_embedding_metadata
from model_registry import get_bundle_by_id, make_model_key
from ha_client import call_ha_service


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _release_process_memory() -> None:
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _percent_from_progress(p: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(p, dict):
        return None
    e = p.get("epoch")
    t = p.get("total_epochs")
    try:
        e = int(e) if e is not None else None
        t = int(t) if t is not None else None
        if not e or not t or t <= 0:
            return None
        return max(0, min(100, int(round((e / t) * 100))))
    except Exception:
        return None


async def _upload_gzip_payload_in_chunks(
    *,
    training_server_url: str,
    api_key: Optional[str],
    gzip_payload_path: str,
    request_id: str,
    chunk_size_bytes: int = 2 * 1024 * 1024,
) -> Dict[str, Any]:
    session_info = await create_training_upload_session(
        training_server_url=training_server_url,
        api_key=api_key,
        timeout_s=30.0,
    )
    upload_id = str(session_info.get("upload_id") or "").strip()
    if not upload_id:
        raise TrainingServerError("Training upload session did not return an upload_id.")

    with open(gzip_payload_path, "rb") as f:
        chunk_index = 0
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            await upload_training_chunk(
                training_server_url=training_server_url,
                api_key=api_key,
                upload_id=upload_id,
                chunk_bytes=chunk,
                chunk_index=chunk_index,
                timeout_s=240.0,
            )
            chunk_index += 1

    return await complete_training_upload_session(
        training_server_url=training_server_url,
        api_key=api_key,
        upload_id=upload_id,
        request_id=request_id,
        timeout_s=1120.0,
    )


async def _start_training_job_with_upload_fallback(
    *,
    training_server_url: str,
    api_key: Optional[str],
    gzip_payload_path: str,
    request_id: str,
    compressed_size_bytes: int,
    n_embeddings: int,
    embedding_dim: int,
) -> Dict[str, Any]:
    try:
        return await _upload_gzip_payload_in_chunks(
            training_server_url=training_server_url,
            api_key=api_key,
            gzip_payload_path=gzip_payload_path,
            request_id=request_id,
        )
    except Exception as exc:
        print(
            f"Chunked training upload failed, falling back to direct gzip POST: {exc}",
            flush=True,
        )
        return await start_training_job_from_gzip_file(
            training_server_url=training_server_url,
            api_key=api_key,
            gzip_payload_path=gzip_payload_path,
            compressed_size_bytes=compressed_size_bytes,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            timeout_s=1120.0,
        )


def _sample_replay_payload(
    embeddings: Any,
    targets_on: Any,
    *,
    max_points: int = 12000,
) -> Dict[str, Any]:
    embeddings_arr = np.asarray(embeddings or [], dtype=np.float32)
    targets_arr = np.asarray(targets_on or [], dtype=np.int32).reshape(-1)
    if embeddings_arr.ndim != 2 or embeddings_arr.shape[0] == 0 or embeddings_arr.shape[0] != targets_arr.size:
        return {
            "embeddings": [],
            "targets_on": [],
            "sampled": False,
            "sample_count": 0,
            "original_count": int(targets_arr.size),
        }

    total = int(targets_arr.size)
    if total <= max_points:
        return {
            "embeddings": embeddings_arr.tolist(),
            "targets_on": targets_arr.tolist(),
            "sampled": False,
            "sample_count": total,
            "original_count": total,
        }

    on_idx = np.flatnonzero(targets_arr == 1)
    off_idx = np.flatnonzero(targets_arr == 0)
    target_on = min(on_idx.size, max_points // 2)
    target_off = min(off_idx.size, max_points - target_on)
    remaining = max_points - (target_on + target_off)
    if remaining > 0:
        if on_idx.size - target_on >= off_idx.size - target_off:
            target_on = min(on_idx.size, target_on + remaining)
        else:
            target_off = min(off_idx.size, target_off + remaining)

    on_sample = np.linspace(0, max(0, on_idx.size - 1), num=target_on, dtype=np.int64) if target_on > 0 else np.zeros((0,), dtype=np.int64)
    off_sample = np.linspace(0, max(0, off_idx.size - 1), num=target_off, dtype=np.int64) if target_off > 0 else np.zeros((0,), dtype=np.int64)
    sampled_idx = np.concatenate([on_idx[on_sample], off_idx[off_sample]])
    sampled_idx.sort()

    sampled_embeddings = embeddings_arr[sampled_idx]
    sampled_targets = targets_arr[sampled_idx]
    return {
        "embeddings": sampled_embeddings.tolist(),
        "targets_on": sampled_targets.tolist(),
        "sampled": True,
        "sample_count": int(sampled_targets.size),
        "original_count": total,
    }


def _binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return 0.0
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    denom = precision + recall
    return float(0.0 if denom <= 0 else (2.0 * precision * recall) / denom)


def _best_prob_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    if y_true.size == 0 or y_prob.size == 0 or y_true.size != y_prob.size or np.unique(y_true).size < 2:
        return 0.5, 0.0

    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= float(thr)).astype(np.int32)
        f1 = _binary_f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return float(best_thr), float(best_f1)


def _summarize_scores(scores: np.ndarray, threshold: float, y_true: np.ndarray) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    if scores.size == 0:
        return {
            "n_points": 0,
            "max_score": None,
            "min_score": None,
            "mean_score": None,
            "threshold": float(threshold),
            "n_above_threshold": 0,
            "f1_at_threshold": 0.0,
        }
    y_pred = (scores >= float(threshold)).astype(np.int32)
    return {
        "n_points": int(scores.size),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "mean_score": float(np.mean(scores)),
        "threshold": float(threshold),
        "n_above_threshold": int(np.sum(y_pred)),
        "f1_at_threshold": _binary_f1_score(y_true, y_pred),
    }


class TrainingServerServiceManager:
    def __init__(
        self,
        *,
        jobs_dir: str,
        models_root: str,
        training_server_url: str,
        training_server_api_key: Optional[str],
        save_embedding_npy_fn,
        reload_algorithm_fn=None,
    ):
        self.jobs_dir = jobs_dir
        self.models_root = models_root
        self.training_server_url = training_server_url
        self.training_server_api_key = training_server_api_key
        self.save_embedding_npy = save_embedding_npy_fn
        self.reload_algorithm = reload_algorithm_fn

        os.makedirs(self.jobs_dir, exist_ok=True)

        # Prevent concurrent read/write races between UI status calls and poller tasks
        self._io_lock = asyncio.Lock()

    async def _notify_training_result(
        self,
        *,
        job_id: str,
        title: str,
        message: str,
    ) -> None:
        if not app_state.TOKEN:
            return
        try:
            await call_ha_service(
                app_state.HA_REST_API_URL,
                app_state.TOKEN,
                "persistent_notification",
                "create",
                {
                    "title": title,
                    "message": message,
                    "notification_id": f"nilm_training_{job_id}",
                },
            )
        except Exception as exc:
            print(f"Failed to create Home Assistant training notification for job_id={job_id}: {exc}", flush=True)

    async def _maybe_reload_algorithm(self) -> None:
        if self.reload_algorithm is None:
            return
        result = self.reload_algorithm()
        if inspect.isawaitable(result):
            await result

    def _job_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _replay_input_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.replay.json")

    def _payload_gzip_path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.payload.json.gz")

    def _read_unlocked(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _read(self, job_id: str) -> Dict[str, Any]:
        path = self._job_path(job_id)
        async with self._io_lock:
            return self._read_unlocked(path)

    async def _write(self, job_id: str, data: Dict[str, Any]) -> None:
        path = self._job_path(job_id)
        async with self._io_lock:
            _atomic_write_json(path, data)

    async def _write_payload_gzip(self, job_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        path = self._payload_gzip_path(job_id)
        tmp = f"{path}.tmp"
        async with self._io_lock:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, path)
        compressed_size = os.path.getsize(path)
        embeddings = data.get("embeddings") if isinstance(data.get("embeddings"), list) else []
        first_embedding = embeddings[0] if embeddings and isinstance(embeddings[0], list) else []
        return {
            "payload_gzip_path": path,
            "payload_gzip_bytes": compressed_size,
            "payload_summary": summarize_training_server_payload(data),
            "embedding_dim": len(first_embedding),
        }

    async def _delete_payload_gzip(self, job_id: str) -> None:
        path = self._payload_gzip_path(job_id)
        async with self._io_lock:
            if os.path.exists(path):
                os.remove(path)

    async def _patch_job(self, job_id: str, patch: Dict[str, Any]) -> None:
        path = self._job_path(job_id)
        async with self._io_lock:
            data = self._read_unlocked(path)
            data.setdefault("_job", {})
            data["_job"] = {**data["_job"], **patch}
            _atomic_write_json(path, data)

    async def _prune_heavy_job_fields(self, job_id: str) -> None:
        path = self._job_path(job_id)
        async with self._io_lock:
            data = self._read_unlocked(path)
            for key in ("embeddings", "targets_on", "targets_power", "weak_mains", "timestamps", "power_windows"):
                if key in data:
                    data.pop(key, None)
            _atomic_write_json(path, data)

    async def _write_replay_input(self, job_id: str, data: Dict[str, Any]) -> None:
        path = self._replay_input_path(job_id)
        async with self._io_lock:
            _atomic_write_json(path, data)

    async def _delete_replay_input(self, job_id: str) -> None:
        path = self._replay_input_path(job_id)
        async with self._io_lock:
            if os.path.exists(path):
                os.remove(path)

    async def _run_edge_replay_worker(self, job_id: str) -> Optional[Dict[str, Any]]:
        worker_path = os.path.join(os.path.dirname(__file__), "edge_replay_worker.py")
        replay_input_path = self._replay_input_path(job_id)
        if not os.path.exists(worker_path) or not os.path.exists(replay_input_path):
            return None

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            worker_path,
            replay_input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"edge replay worker failed with code {proc.returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"edge replay worker returned invalid JSON: {exc}") from exc
        return payload if isinstance(payload, dict) else None

    async def create_job(self, prepared: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        training_payload = training_server_payload_from_prepared(prepared)
        payload_meta = await self._write_payload_gzip(job_id, training_payload)
        bundle_id = str(prepared.get("bundle_id") or "").strip()
        bundle = get_bundle_by_id(app_state.model_bundles, bundle_id) if bundle_id else None
        if bundle is not None:
            replay_sample = _sample_replay_payload(prepared.get("embeddings"), prepared.get("targets_on"))
            replay_input = {
                "job_id": job_id,
                "appliance_name": str(prepared.get("appliance_name") or ""),
                "bundle_id": bundle_id,
                "inference_dir": bundle.inference_dir,
                "embeddings_dir": bundle_models_dir(self.models_root, bundle_id),
                "embeddings": replay_sample["embeddings"],
                "targets_on": replay_sample["targets_on"],
                "sampled": replay_sample["sampled"],
                "sample_count": replay_sample["sample_count"],
                "original_count": replay_sample["original_count"],
            }
            await self._write_replay_input(job_id, replay_input)

        job_record = {
            "appliance_name": prepared.get("appliance_name"),
            "appliance_type": prepared.get("appliance_type"),
            "supervision_mode": prepared.get("supervision_mode"),
            "appliance_sensor_id": prepared.get("appliance_sensor_id"),
            "main_sensor_id": prepared.get("main_sensor_id"),
            "bundle_id": prepared.get("bundle_id"),
            "bundle_mode": prepared.get("bundle_mode"),
            "bundle_version": prepared.get("bundle_version"),
            "settings": prepared.get("settings") or {},
            "stats": prepared.get("stats") or {},
            "payload_gzip_path": payload_meta["payload_gzip_path"],
            "payload_gzip_bytes": payload_meta["payload_gzip_bytes"],
            "payload_summary": payload_meta["payload_summary"],
            "_job": {
                "job_id": job_id,
                "state": "prepared",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "training_server_job_id": None,
                "saved_path": None,
                "embedding_dim": None,
                "error": None,
                "progress": {
                    "phase": "prepared",
                    "epoch": 0,
                    "total_epochs": None,
                    "loss": None,
                    "min_loss": None,
                },
                "percent": None,
            },
        }
        await self._write(job_id, job_record)
        print(
            f"Created local training job local_job_id={job_id} "
            f"appliance={prepared.get('appliance_name')} bundle_id={prepared.get('bundle_id')} "
            f"n_embeddings={payload_meta['payload_summary'].get('n_embeddings')}",
            flush=True,
        )
        return job_id

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        data = await self._read(job_id)
        j = data.get("_job", {}) or {}
        return {
            "status": "success",
            "job_id": job_id,
            "state": j.get("state", "unknown"),
            "training_server_job_id": j.get("training_server_job_id"),
            "saved_path": j.get("saved_path"),
            "embedding_dim": j.get("embedding_dim"),
            "error": j.get("error"),
            "progress": j.get("progress"),
            "percent": j.get("percent"),
            "training_metrics": j.get("training_metrics"),
            "updated_at": j.get("updated_at"),
        }

    async def get_training_server_connection_status(self) -> Dict[str, Any]:
        url_state = await app_state.resolve_training_server_url_state()
        training_server_url = url_state["effective_training_server_url"]
        if not training_server_url:
            autodetect = url_state.get("autodetect") or {}
            message = "Training server URL is not configured."
            if autodetect.get("ok") and autodetect.get("training_server_url"):
                message = (
                    "Training server is not selected yet. "
                    "Select the detected server to use it."
                )
            elif autodetect.get("message"):
                message = f"{message} {autodetect.get('message')}"
            return {
                "status": "success",
                "ok": False,
                "state": "missing_config",
                "message": message,
                "training_server_url": "",
                **url_state,
            }
        training_server_api_key = app_state.get_training_server_api_key()
        result = await probe_training_server_connection(
            training_server_url=training_server_url,
            api_key=training_server_api_key,
            timeout_s=8.0,
        )
        return {
            "status": "success",
            "training_server_url": training_server_url,
            **url_state,
            **result,
        }

    async def detect_training_server(self) -> Dict[str, Any]:
        url_state = await app_state.resolve_training_server_url_state()
        autodetect = url_state.get("autodetect") or {}
        return {
            "status": "success",
            **autodetect,
            "configured_training_server_url": url_state.get("configured_training_server_url", ""),
            "effective_training_server_url": url_state.get("effective_training_server_url", ""),
            "training_server_url_source": url_state.get("training_server_url_source"),
        }

    async def start_send(self, job_id: str) -> Dict[str, Any]:
        prepared = await self._read(job_id)
        payload_summary = prepared.get("payload_summary") or {}
        payload_gzip_path = str(prepared.get("payload_gzip_path") or "").strip()
        payload_gzip_bytes = int(prepared.get("payload_gzip_bytes") or 0)
        url_state = await app_state.resolve_training_server_url_state()
        training_server_url = url_state["effective_training_server_url"]
        training_server_api_key = app_state.get_training_server_api_key()
        print(
            f"Starting training server send for local_job_id={job_id} "
            f"url={training_server_url}: {json.dumps(payload_summary, sort_keys=True)}",
            flush=True,
        )

        start = await _start_training_job_with_upload_fallback(
            training_server_url=training_server_url,
            api_key=training_server_api_key,
            gzip_payload_path=payload_gzip_path,
            request_id=str(prepared.get("_job", {}).get("job_id") or job_id),
            compressed_size_bytes=payload_gzip_bytes,
            n_embeddings=int(payload_summary.get("n_embeddings") or 0),
            embedding_dim=int(payload_summary.get("embedding_dim") or 0),
        )

        training_server_job_id = start["job_id"]
        print(
            f"Training server accepted local_job_id={job_id} training_server_job_id={training_server_job_id}",
            flush=True,
        )

        await self._patch_job(job_id, {
            "training_server_job_id": training_server_job_id,
            "training_server_url": training_server_url,
            "training_server_url_source": url_state.get("training_server_url_source"),
            "state": "training_server_queued",
            "updated_at": _utc_now_iso(),
            "error": None,
            "progress": {
                "phase": "queued",
                "epoch": 0,
                "total_epochs": None,
                "loss": None,
                "min_loss": None,
            },
            "percent": None,
        })

        # background poller updates progress + finalizes
        asyncio.create_task(self._poll_and_finalize(job_id, training_server_job_id, training_server_url, training_server_api_key))

        del prepared
        _release_process_memory()
        return {"status": "success", "job_id": job_id, "training_server_job_id": training_server_job_id}

    async def _merge_training_server_status(self, job_id: str, st: Dict[str, Any]) -> None:
        """
        st is the raw JSON from the training server GET /train/{job_id}
        """
        training_server_status = (st.get("status") or "").lower()  # queued|running|done|error
        progress = st.get("progress")
        percent = _percent_from_progress(progress)

        # Map training server status -> app state
        if training_server_status == "queued":
            state = "training_server_queued"
        elif training_server_status == "running":
            state = "training_server_running"
        elif training_server_status == "done":
            state = "training_server_done"
        elif training_server_status == "error":
            state = "error"
        else:
            state = "training_server_running"  # fallback

        patch: Dict[str, Any] = {
            "state": state,
            "updated_at": _utc_now_iso(),
        }

        if isinstance(progress, dict):
            patch["progress"] = progress
            patch["percent"] = percent

        if training_server_status == "error":
            patch["error"] = st.get("message") or st.get("error") or "Training server error"

        await self._patch_job(job_id, patch)

    async def _poll_and_finalize(
        self,
        job_id: str,
        training_server_job_id: str,
        training_server_url: str,
        training_server_api_key: Optional[str],
    ) -> None:
        try:
            # 1) Kick off a lightweight status loop to keep progress fresh
            #    (even if poll_training_result sleeps / waits long)
            async def progress_loop():
                while True:
                    st = await fetch_training_status(
                        training_server_url=training_server_url,
                        api_key=training_server_api_key,
                        job_id=training_server_job_id,
                        timeout_s=240.0,
                    )
                    await self._merge_training_server_status(job_id, st)

                    s = (st.get("status") or "").lower()
                    if s in ("done", "error"):
                        return
                    await asyncio.sleep(2.0)

            progress_task = asyncio.create_task(progress_loop())

            # 2) Wait for final result (this returns only when DONE)
            result = await poll_training_result(
                training_server_url=training_server_url,
                api_key=training_server_api_key,
                job_id=training_server_job_id,
                poll_every_s=2.0,
                max_wait_s=1800.0,
            )

            # ensure progress loop stops
            try:
                await progress_task
            except Exception:
                pass

            embedding = result.get("embedding")
            appliance_name = (
                result.get("appliance_name")
                or (await self._read(job_id)).get("appliance_name")
                or (await self._read(job_id)).get("_meta", {}).get("appliance_name")
            )

            if not appliance_name or not isinstance(embedding, list) or not embedding:
                raise RuntimeError("Training server result missing appliance_name/embedding")

            prepared_job = await self._read(job_id)
            bundle_id = str(
                result.get("bundle_id")
                or prepared_job.get("bundle_id")
                or prepared_job.get("_meta", {}).get("bundle_id")
                or ""
            ).strip()
            if not bundle_id:
                raise RuntimeError("Training server result missing bundle_id")

            bundle_dir = bundle_models_dir(self.models_root, bundle_id)
            existing_metadata = load_embedding_metadata(bundle_dir, str(appliance_name)) or {}
            saved_path = self.save_embedding_npy(bundle_dir, str(appliance_name), embedding)
            trainer_onoff_threshold = float(result.get("appliance_params", {}).get("onoff_threshold", 0.5))
            trainer_onoff_f1 = result.get("stats", {}).get("onoff_f1")
            deployment_onoff_threshold = trainer_onoff_threshold
            deployment_onoff_f1 = trainer_onoff_f1

            edge_replay_metrics = None
            try:
                edge_replay_metrics = await self._run_edge_replay_worker(job_id)
                if edge_replay_metrics:
                    deployment_onoff_threshold = float(edge_replay_metrics.get("threshold", trainer_onoff_threshold))
                    deployment_onoff_f1 = float(edge_replay_metrics.get("f1_at_threshold", trainer_onoff_f1 or 0.0))
                    edge_replay_metrics["trainer_threshold"] = float(trainer_onoff_threshold)
                    edge_replay_metrics["trainer_onoff_f1"] = None if trainer_onoff_f1 is None else float(trainer_onoff_f1)
                    edge_replay_metrics["deployment_threshold"] = float(deployment_onoff_threshold)
                    edge_replay_metrics["deployment_onoff_f1"] = None if deployment_onoff_f1 is None else float(deployment_onoff_f1)
                    print(
                        f"Edge replay summary appliance={appliance_name} "
                        f"n_points={edge_replay_metrics.get('n_points')} "
                        f"max_score={edge_replay_metrics.get('max_score')} "
                        f"mean_score={edge_replay_metrics.get('mean_score')} "
                        f"threshold={edge_replay_metrics.get('threshold')} "
                        f"n_above_threshold={edge_replay_metrics.get('n_above_threshold')} "
                        f"f1_at_threshold={edge_replay_metrics.get('f1_at_threshold')}",
                        flush=True,
                    )
            except Exception as replay_exc:
                print(f"Edge replay summary failed for appliance={appliance_name}: {replay_exc}", flush=True)
                print(traceback.format_exc(), flush=True)
            finally:
                await self._delete_replay_input(job_id)
                _release_process_memory()

            save_embedding_metadata(
                bundle_dir,
                str(appliance_name),
                {
                    **existing_metadata,
                    "appliance_name": str(appliance_name),
                    "bundle_id": bundle_id,
                    "bundle_mode": prepared_job.get("bundle_mode") or result.get("bundle_mode"),
                    "bundle_version": prepared_job.get("bundle_version"),
                    "saved_path": saved_path,
                    "saved_at": _utc_now_iso(),
                    "stats": prepared_job.get("stats", {}) or {},
                    "supervision_mode": prepared_job.get("supervision_mode"),
                    "appliance_sensor_id": prepared_job.get("appliance_sensor_id"),
                    "main_sensor_id": prepared_job.get("main_sensor_id"),
                    "job_id": job_id,
                    "training_server_job_id": training_server_job_id,
                    "onoff_threshold": float(deployment_onoff_threshold),
                    "power_threshold": float(result.get("appliance_params", {}).get("power_threshold", 0.0)),
                    "onoff_f1": deployment_onoff_f1,
                    "power_f1": result.get("stats", {}).get("power_f1"),
                    "fine_tune_target": result.get("stats", {}).get("fine_tune_target"),
                    "trainer_onoff_threshold": float(trainer_onoff_threshold),
                    "trainer_onoff_f1": trainer_onoff_f1,
                },
            )
            app_state.set_model_mains_assignment(make_model_key(bundle_id, str(appliance_name)), prepared_job.get("main_sensor_id"))

            await self._maybe_reload_algorithm()

            await self._patch_job(job_id, {
                "state": "done",
                "updated_at": _utc_now_iso(),
                "saved_path": saved_path,
                "embedding_dim": len(embedding),
                "error": None,
                "training_metrics": {
                    "onoff_f1": deployment_onoff_f1,
                    "onoff_threshold": deployment_onoff_threshold,
                    "power_f1": result.get("stats", {}).get("power_f1"),
                    "power_threshold": result.get("appliance_params", {}).get("power_threshold"),
                    "fine_tune_target": result.get("stats", {}).get("fine_tune_target"),
                    "trainer_onoff_f1": trainer_onoff_f1,
                    "trainer_onoff_threshold": trainer_onoff_threshold,
                    "edge_replay": edge_replay_metrics,
                },
                "progress": {
                    "phase": "done",
                    "epoch": len(embedding) and prepared_job.get("_job", {}).get("progress", {}).get("epoch"),
                    "total_epochs": prepared_job.get("_job", {}).get("progress", {}).get("total_epochs"),
                    "loss": prepared_job.get("_job", {}).get("progress", {}).get("loss"),
                    "min_loss": prepared_job.get("_job", {}).get("progress", {}).get("min_loss"),
                },
                "percent": 100,
            })
            await self._delete_payload_gzip(job_id)
            del prepared_job
            _release_process_memory()
            await self._notify_training_result(
                job_id=job_id,
                title=f"NILM training finished: {appliance_name}",
                message=f"Training completed successfully for {appliance_name}. The appliance model is now available for disaggregation in the NILM Dashboard.",
            )
            print(
                f"Training finalize done local_job_id={job_id} training_server_job_id={training_server_job_id} "
                f"saved_path={saved_path}",
                flush=True,
            )

        except Exception as e:
            print(
                f"Training finalize failed local_job_id={job_id} training_server_job_id={training_server_job_id}: {e}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            await self._patch_job(job_id, {
                "state": "error",
                "updated_at": _utc_now_iso(),
                "error": str(e),
                "progress": {"phase": "error"},
            })
            await self._delete_payload_gzip(job_id)
            await self._notify_training_result(
                job_id=job_id,
                title="NILM training failed",
                message="\n".join([
                    f"Training failed for job `{job_id}`.",
                    "",
                    f"Error: {e}",
                    "",
                    "Open the Appliance Training Session page to review the job and logs.",
                ]),
            )
            _release_process_memory()
