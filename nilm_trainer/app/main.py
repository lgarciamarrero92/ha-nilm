from __future__ import annotations

import asyncio
import gzip
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.losses import resolve_losses
from app.model_registry import discover_server_bundles, get_bundle_by_id, get_latest_bundle_for_mode
from app.trainer import train_ref_embedding

config_path = Path(__file__).resolve().parents[1] / "config.yaml"
VERSION = "unknown"
if config_path.is_file():
    with config_path.open(encoding="utf-8") as config:
        VERSION = str(yaml.safe_load(config)["version"])

app = FastAPI(version=VERSION)

# NOTE: single-worker only. For multi-worker, use Redis/DB.
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = asyncio.Lock()
UPLOADS: Dict[str, Dict[str, Any]] = {}
UPLOADS_LOCK = asyncio.Lock()
BUNDLES_ROOT = "/app/bundles"
SERVER_BUNDLES = discover_server_bundles(BUNDLES_ROOT)
UPLOADS_ROOT = os.path.join(tempfile.gettempdir(), "nilm_training_uploads")
os.makedirs(UPLOADS_ROOT, exist_ok=True)
# --------------------------
# SERVER-OWNED TRAINING CONFIG (client cannot change)
# --------------------------
SERVER_TRAINING = {
    "epochs": 200,
    "lr": 0.01,
    "batch_size": 1024,
    "seed": 0,
    "patience": 5,
    "min_delta": 1e-4,
}
LOSS_PROFILE = {
    "onoff": {"name": "binary_crossentropy"},
    "power": {"name": "mse"},
}
LOSS_PROFILE.update(json.loads(os.environ.get("NILM_TRAINING_LOSSES", "{}")))
TRAINING_LOSSES = resolve_losses(LOSS_PROFILE)

class TrainPayload(BaseModel):
    appliance_name: str
    appliance_type: str = ""
    embeddings: List[List[float]]
    targets_on: List[int]
    targets_power: Optional[List[float]] = None
    weak_mains: Optional[List[float]] = None
    supervision_mode: Optional[str] = None
    appliance_sensor_id: Optional[str] = None
    bundle_id: Optional[str] = None
    bundle_mode: Optional[str] = None
    bundle_version: Optional[int] = None
    settings: Dict[str, Any]


class UploadCompletePayload(BaseModel):
    request_id: Optional[str] = None


async def _accept_training_payload(payload: TrainPayload, *, client_host: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    request_id = request_id or uuid.uuid4().hex[:8]
    job_id = uuid.uuid4().hex

    n_emb = len(payload.embeddings)
    n_y = len(payload.targets_on)

    if n_emb == 0 or n_y == 0 or n_emb != n_y:
        raise HTTPException(status_code=400, detail="Invalid dataset sizes (embeddings/targets).")
    if payload.targets_power is not None and len(payload.targets_power) != n_emb:
        raise HTTPException(status_code=400, detail="Invalid dataset sizes (embeddings/targets_power).")
    if payload.weak_mains is not None and len(payload.weak_mains) != n_emb:
        raise HTTPException(status_code=400, detail="Invalid dataset sizes (embeddings/weak_mains).")

    selected_bundle = None
    if payload.bundle_id:
        selected_bundle = get_bundle_by_id(SERVER_BUNDLES, payload.bundle_id)
        if selected_bundle is None:
            raise HTTPException(status_code=400, detail=f"Unknown bundle_id: {payload.bundle_id}")
    else:
        selected_bundle = get_latest_bundle_for_mode(SERVER_BUNDLES, payload.bundle_mode or "online")
        if selected_bundle is None:
            raise HTTPException(status_code=400, detail=f"No server bundle available for mode: {payload.bundle_mode or 'online'}")

    loop = asyncio.get_running_loop()
    now = time.time()
    async with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "request_id": request_id,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "appliance_name": payload.appliance_name,
            "appliance_type": payload.appliance_type,
            "supervision_mode": payload.supervision_mode,
            "bundle_id": selected_bundle.bundle_id,
            "bundle_mode": selected_bundle.mode,
            "bundle_version": selected_bundle.model_version,
            "n_samples": n_emb,
            "progress": {
                "phase": "queued",
                "epoch": 0,
                "total_epochs": int(SERVER_TRAINING["epochs"]),
                "loss": None,
                "min_loss": None,
            },
        }

    print(
        f"[{request_id}] /train accepted job_id={job_id} from {client_host} "
        f"n={n_emb} bundle={selected_bundle.bundle_id} "
        f"(epochs={SERVER_TRAINING['epochs']} bs={SERVER_TRAINING['batch_size']})",
        flush=True,
    )

    asyncio.create_task(_run_training_job(job_id, payload, selected_bundle, request_id, loop))
    return {"status": "accepted", "job_id": job_id, "request_id": request_id}


def _connection_info() -> Dict[str, str]:
    addon_hostname = (os.getenv("HOSTNAME") or "").strip()
    training_server_url = f"http://{addon_hostname}:8080/train" if addon_hostname else ""
    return {
        "hostname": addon_hostname,
        "training_server_url": training_server_url,
    }


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    info = _connection_info()
    hostname = info["hostname"] or "unavailable"
    training_server_url = info["training_server_url"] or "unavailable"
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>NILM Training Server</title>
      <style>
        body {{
          margin: 0;
          font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
          color: #0f172a;
        }}
        main {{
          max-width: 760px;
          margin: 48px auto;
          padding: 28px;
          background: rgba(255, 255, 255, 0.95);
          border: 1px solid #dbeafe;
          border-radius: 20px;
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }}
        h1 {{
          margin-top: 0;
          margin-bottom: 10px;
        }}
        p {{
          color: #475569;
          line-height: 1.6;
        }}
        .card {{
          margin-top: 20px;
          padding: 16px;
          border-radius: 16px;
          border: 1px solid #bfdbfe;
          background: #f8fbff;
        }}
        .label {{
          font-size: 0.85rem;
          font-weight: 700;
          color: #1d4ed8;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          margin-bottom: 6px;
        }}
        code {{
          display: block;
          padding: 12px 14px;
          border-radius: 12px;
          background: #0f172a;
          color: #e2e8f0;
          overflow-wrap: anywhere;
          font-size: 0.95rem;
        }}
      </style>
    </head>
    <body>
      <main>
        <h1>NILM Training Server</h1>
        <p>Use the value below in the <strong>NILM</strong> app Configuration tab as <code style="display:inline;padding:2px 6px;background:#e2e8f0;color:#0f172a;">training_server_url</code>.</p>
        <div class="card">
          <div class="label">Hostname</div>
          <code>{hostname}</code>
        </div>
        <div class="card">
          <div class="label">Training Server URL</div>
          <code>{training_server_url}</code>
        </div>
      </main>
    </body>
    </html>
    """


@app.get("/connection-info")
async def connection_info():
    return _connection_info()

@app.get("/version")
def version():
    return VERSION

@app.post("/train", status_code=202)
async def start_train(request: Request):
    content_encoding = (request.headers.get("content-encoding") or "").strip().lower()
    raw_body = await request.body()
    print(
        f"/train body received bytes={len(raw_body)} content_encoding={content_encoding or 'identity'}",
        flush=True,
    )

    try:
        if content_encoding == "gzip":
            decoded_body = gzip.decompress(raw_body)
        else:
            decoded_body = raw_body

        payload_obj = json.loads(decoded_body.decode("utf-8")) if decoded_body else {}
        payload = TrainPayload.parse_obj(payload_obj)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid training payload: {exc}")

    client_host = request.client.host if request.client else "unknown"
    request_id = uuid.uuid4().hex[:8]
    return await _accept_training_payload(payload, client_host=client_host, request_id=request_id)


@app.post("/train/uploads", status_code=201)
async def create_train_upload():
    upload_id = uuid.uuid4().hex
    upload_path = os.path.join(UPLOADS_ROOT, f"{upload_id}.json.gz")
    async with UPLOADS_LOCK:
        UPLOADS[upload_id] = {
            "upload_id": upload_id,
            "path": upload_path,
            "created_at": time.time(),
            "updated_at": time.time(),
            "chunks_received": 0,
            "bytes_received": 0,
        }
    return {"status": "created", "upload_id": upload_id}


@app.post("/train/uploads/{upload_id}/chunk", status_code=202)
async def append_train_upload_chunk(upload_id: str, request: Request):
    chunk_index_raw = request.query_params.get("chunk_index")
    try:
        chunk_index = int(chunk_index_raw) if chunk_index_raw is not None else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="chunk_index must be an integer")

    raw_chunk = await request.body()
    if not raw_chunk:
        raise HTTPException(status_code=400, detail="Empty upload chunk")

    async with UPLOADS_LOCK:
        upload = UPLOADS.get(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="upload_id not found")
        expected_chunk_index = int(upload.get("chunks_received") or 0)
        if chunk_index is not None and chunk_index != expected_chunk_index:
            raise HTTPException(status_code=409, detail=f"Expected chunk_index {expected_chunk_index}, got {chunk_index}")
        with open(upload["path"], "ab") as f:
            f.write(raw_chunk)
        upload["chunks_received"] = expected_chunk_index + 1
        upload["bytes_received"] = int(upload.get("bytes_received") or 0) + len(raw_chunk)
        upload["updated_at"] = time.time()
        return {
            "status": "accepted",
            "upload_id": upload_id,
            "chunk_index": expected_chunk_index,
            "chunks_received": upload["chunks_received"],
            "bytes_received": upload["bytes_received"],
        }


@app.post("/train/uploads/{upload_id}/complete", status_code=202)
async def complete_train_upload(upload_id: str, body: UploadCompletePayload, request: Request):
    async with UPLOADS_LOCK:
        upload = UPLOADS.get(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="upload_id not found")
        upload_path = str(upload.get("path") or "")

    if not upload_path or not os.path.exists(upload_path):
        raise HTTPException(status_code=404, detail="Uploaded payload file not found")

    try:
        with gzip.open(upload_path, "rt", encoding="utf-8") as gz:
            payload_obj = json.load(gz)
        payload = TrainPayload.parse_obj(payload_obj)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid uploaded training payload: {exc}")
    finally:
        async with UPLOADS_LOCK:
            upload = UPLOADS.pop(upload_id, None)
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass

    client_host = request.client.host if request.client else "unknown"
    return await _accept_training_payload(payload, client_host=client_host, request_id=body.request_id)


@app.get("/train/{job_id}", status_code=200)
async def train_status(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job_id not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "request_id": job.get("request_id"),
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "n_samples": job.get("n_samples"),
        "progress": job.get("progress"),
    }


@app.get("/train/{job_id}/result", status_code=200)
async def train_result(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job_id not found")

    if job["status"] == "error":
        raise HTTPException(500, job.get("message", "training failed"))
    if job["status"] != "done":
        raise HTTPException(409, f"not ready ({job['status']})")

    return {
        "status": "success",
        "job_id": job_id,
        "appliance_name": job["result"]["appliance_name"],
        "embedding": job["result"]["embedding"],
        "embedding_dim": job["result"]["embedding_dim"],
        "bundle_id": job["result"].get("bundle_id"),
        "bundle_mode": job["result"].get("bundle_mode"),
        "bundle_version": job["result"].get("bundle_version"),
        "appliance_params": job["result"].get("appliance_params", {}),
        "stats": job["result"]["stats"],
    }


async def _run_training_job(job_id: str, payload: TrainPayload, selected_bundle, request_id: str, loop: asyncio.AbstractEventLoop):
    training_started_mono = time.perf_counter()
    async with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return
        j["status"] = "running"
        j["started_at"] = time.time()
        j["updated_at"] = time.time()
        j["progress"]["phase"] = "running"

    def push_progress(p: Dict[str, Any]) -> None:
        """Called from training thread; schedules an async merge into JOBS."""
        async def _merge():
            async with JOBS_LOCK:
                jj = JOBS.get(job_id)
                if not jj:
                    return
                jj["updated_at"] = time.time()
                jj["progress"] = {**jj.get("progress", {}), **p}

        loop.call_soon_threadsafe(lambda: asyncio.create_task(_merge()))

    try:
        embedding, stats, appliance_params = await asyncio.to_thread(
            train_ref_embedding,
            query_embeddings=payload.embeddings,
            targets_on=payload.targets_on,
            targets_power=payload.targets_power,
            weak_mains=payload.weak_mains,
            settings=payload.settings,
            load_head=selected_bundle.load_head,
            losses=TRAINING_LOSSES,
            epochs=int(SERVER_TRAINING["epochs"]),
            lr=float(SERVER_TRAINING["lr"]),
            batch_size=int(SERVER_TRAINING["batch_size"]),
            seed=int(SERVER_TRAINING["seed"]),
            patience=int(SERVER_TRAINING["patience"]),
            min_delta=float(SERVER_TRAINING["min_delta"]),
            on_progress=push_progress,
        )
        stats = {**stats, "loss_profile": LOSS_PROFILE}

        async with JOBS_LOCK:
            jj = JOBS.get(job_id)
            if not jj:
                return
            jj["status"] = "done"
            jj["updated_at"] = time.time()
            jj["progress"] = {
                **jj.get("progress", {}),
                "phase": "done",
                "epoch": stats.get("epochs_ran"),
                "total_epochs": stats.get("epochs_requested"),
                "loss": stats.get("final_loss"),
                "min_loss": stats.get("min_loss"),
                "cls_loss": stats.get("final_cls_loss"),
                "reg_loss": stats.get("final_reg_loss"),
                "fine_tune_target": stats.get("fine_tune_target"),
            }
            jj["result"] = {
                "appliance_name": payload.appliance_name,
                "bundle_id": selected_bundle.bundle_id,
                "bundle_mode": selected_bundle.mode,
                "bundle_version": selected_bundle.model_version,
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "appliance_params": appliance_params,
                "stats": stats,
            }

        training_duration_s = time.perf_counter() - training_started_mono
        print(
            f"[{request_id}] job_id={job_id} DONE "
            f"embedding_dim={len(embedding)} training_duration_s={training_duration_s:.3f}",
            flush=True,
        )

    except Exception as e:
        async with JOBS_LOCK:
            jj = JOBS.get(job_id)
            if not jj:
                return
            jj["status"] = "error"
            jj["updated_at"] = time.time()
            jj["message"] = str(e)
            jj["progress"] = {**jj.get("progress", {}), "phase": "error"}

        print(f"[{request_id}] job_id={job_id} ERROR: {e}", flush=True)
