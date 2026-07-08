import json
import traceback

from aiohttp import web

import app_state
from app_state import maybe_await
from model_registry import get_bundle_by_id, get_latest_bundle_for_mode
from prepare_training_data import (
    build_embeddings_training_payload,
    build_uniform_grid,
    compute_on_mask,
    load_model_settings,
    parse_mains_points,
    parse_power_points,
    zoh_resample_to_grid,
)
from runtime_settings import DEFAULT_SENSOR_MAX_GAP_S


def _build_sensor_supervision_intervals(
    *,
    full_history,
    appliance_sensor_history,
    inference_dir: str,
    align_grid: str = "start",
    max_hold_factor: float = 5.0,
    max_hold_s: float = DEFAULT_SENSOR_MAX_GAP_S,
    on_threshold_w: float = 20.0,
    min_on_s: float = 60.0,
    min_off_s: float = 300.0,
):
    settings = load_model_settings(f"{inference_dir}/model_settings.json")
    pts = parse_mains_points(full_history)
    gt_pts = parse_power_points(appliance_sensor_history)

    if len(pts) < 2:
        raise ValueError("Not enough valid mains points after cleaning.")
    if len(gt_pts) < 2:
        raise ValueError("applianceSensorHistoryData must contain at least two valid points for sensor supervision.")

    dt = settings.frequency_s
    t_start = pts[0][0]
    t_end = pts[-1][0]
    grid_t = build_uniform_grid(t_start, t_end, dt, align=align_grid)
    if grid_t.size == 0:
        return []

    max_hold_s = float(max_hold_s) if max_hold_s and max_hold_s > 0 else (
        float(max_hold_factor) * dt if max_hold_factor and max_hold_factor > 0 else None
    )
    gt_grid = zoh_resample_to_grid(gt_pts, grid_t, max_hold_s=max_hold_s, fill_value=0.0)
    gt_grid = gt_grid.astype("float32")
    gt_grid[gt_grid < 0.0] = 0.0

    gt_on_mask = compute_on_mask(
        gt_grid,
        sample_period_s=dt,
        threshold_watts=float(on_threshold_w),
        window_hours=24.0,
        min_on_s=float(min_on_s),
        min_off_s=float(min_off_s),
    )

    intervals = []
    run_start = None
    dt_ms = int(round(float(dt) * 1000.0))
    for idx, bit in enumerate(gt_on_mask.tolist()):
        if bit == 1 and run_start is None:
            run_start = idx
        elif bit == 0 and run_start is not None:
            start_ms = int(round(float(grid_t[run_start]) * 1000.0))
            end_ms = int(round(float(grid_t[idx - 1]) * 1000.0)) + dt_ms
            if end_ms > start_ms:
                intervals.append(
                    {
                        "id": f"sensor_interval_{start_ms}_{end_ms}",
                        "start": start_ms,
                        "end": end_ms,
                    }
                )
            run_start = None

    if run_start is not None:
        start_ms = int(round(float(grid_t[run_start]) * 1000.0))
        end_ms = int(round(float(grid_t[-1]) * 1000.0)) + dt_ms
        if end_ms > start_ms:
            intervals.append(
                {
                    "id": f"sensor_interval_{start_ms}_{end_ms}",
                    "start": start_ms,
                    "end": end_ms,
                }
            )

    return intervals


async def receive_training_data_handler(request: web.Request) -> web.Response:
    try:
        svc = request.app.get("training_server_manager")
        if svc is None:
            return web.json_response({"status": "error", "message": "training_server_manager not configured"}, status=500)

        action = (request.query.get("action") or "").strip().lower()
        if action not in ("prepare", "send", "status", "training_server_status", "training_server_detect", "sensor_intervals"):
            return web.json_response(
                {"status": "error", "message": "Missing/invalid action. Use ?action=prepare|send|status|training_server_status|training_server_detect|sensor_intervals"},
                status=400,
            )

        if action == "prepare":
            data = await request.json()
            if not isinstance(data, dict) or not data:
                return web.json_response({"status": "error", "message": "No JSON body received"}, status=400)

            appliance_name = str(data.get("appliance_name") or "").strip()
            supervision_mode = str(data.get("supervision_mode") or "intervals").strip().lower()
            appliance_sensor_id = str(data.get("appliance_sensor_id") or "").strip() or None
            bundle_id = str(data.get("bundle_id") or "").strip() or None
            bundle_mode = str(data.get("bundle_mode") or "online").strip().lower()
            selected_windows = data.get("selectedWindows")
            full_history = data.get("fullSensorHistoryData")
            appliance_sensor_history = data.get("applianceSensorHistoryData")
            print(
                "Training prepare request "
                f"appliance={appliance_name} supervision_mode={supervision_mode} "
                f"bundle_id={bundle_id} bundle_mode={bundle_mode} "
                f"full_history_points={len(full_history) if isinstance(full_history, list) else 'invalid'} "
                f"selected_windows={len(selected_windows) if isinstance(selected_windows, list) else 'invalid'} "
                f"appliance_history_points={len(appliance_sensor_history) if isinstance(appliance_sensor_history, list) else 'invalid'}",
                flush=True,
            )

            if not appliance_name:
                return web.json_response({"status": "error", "message": "appliance_name is required"}, status=400)
            if not isinstance(full_history, list) or len(full_history) == 0:
                return web.json_response({"status": "error", "message": "fullSensorHistoryData must be a non-empty list"}, status=400)
            if supervision_mode not in ("intervals", "sensor"):
                return web.json_response({"status": "error", "message": "supervision_mode must be 'intervals' or 'sensor'"}, status=400)
            if supervision_mode == "intervals":
                if not isinstance(selected_windows, list) or len(selected_windows) == 0:
                    return web.json_response({"status": "error", "message": "selectedWindows must be a non-empty list for interval supervision"}, status=400)
            else:
                if not appliance_sensor_id:
                    return web.json_response({"status": "error", "message": "appliance_sensor_id is required for sensor supervision"}, status=400)
                if not isinstance(appliance_sensor_history, list) or len(appliance_sensor_history) == 0:
                    return web.json_response({"status": "error", "message": "applianceSensorHistoryData must be a non-empty list for sensor supervision"}, status=400)

            selected_bundle = None
            if bundle_id:
                selected_bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
                if selected_bundle is None:
                    return web.json_response({"status": "error", "message": f"Unknown bundle_id: {bundle_id}"}, status=400)
            else:
                selected_bundle = get_latest_bundle_for_mode(app_state.model_bundles, bundle_mode)

            if selected_bundle is None:
                return web.json_response({"status": "error", "message": f"No {bundle_mode} model bundle is available for training"}, status=400)

            try:
                sensor_max_gap_s = app_state.get_sensor_max_gap_s()
                prepared = build_embeddings_training_payload(
                    fullSensorHistoryData=full_history,
                    selectedWindows=selected_windows,
                    applianceSensorHistoryData=appliance_sensor_history,
                    appliance_name=appliance_name,
                    appliance_type="",
                    supervision_mode=supervision_mode,
                    appliance_sensor_id=appliance_sensor_id,
                    inference_dir=selected_bundle.inference_dir,
                    embeddings_only=True,
                    num_threads=2,
                    batch_size=app_state.get_batch_size(),
                    align_grid="start",
                    max_hold_s=sensor_max_gap_s,
                )
                prepared["bundle_id"] = selected_bundle.bundle_id
                prepared["bundle_mode"] = selected_bundle.mode
                prepared["bundle_version"] = selected_bundle.model_version
                print(
                    "Training prepare completed "
                    f"appliance={appliance_name} bundle={selected_bundle.bundle_id} "
                    f"n_embeddings_ok={prepared.get('stats', {}).get('n_embeddings_ok')} "
                    f"n_windows_total={prepared.get('stats', {}).get('n_windows_total')} "
                    f"n_windows_after_gap_filter={prepared.get('stats', {}).get('n_windows_after_gap_filter')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Training prepare failed: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                return web.json_response({"status": "error", "message": f"Failed to prepare training data: {exc}"}, status=500)

            try:
                job_id = await maybe_await(svc.create_job(prepared))
                print(f"Training job persisted local_job_id={job_id}", flush=True)
            except Exception as exc:
                print(f"Training job persist failed: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                return web.json_response({"status": "error", "message": f"Failed to persist job: {exc}"}, status=500)

            stats = prepared.get("stats", {})
            mode_label = stats.get("supervision_mode", supervision_mode)
            msg = (
                f"Ready! Prepared {stats.get('n_embeddings_ok', '?')} embeddings "
                f"from {stats.get('n_windows_total', '?')} windows using {mode_label} supervision. "
                f"ON fraction: {float(stats.get('on_fraction', 0.0)):.3f}. "
                f"Target bundle: {selected_bundle.label}."
            )
            return web.json_response(
                {
                    "status": "success",
                    "job_id": job_id,
                    "stats": stats,
                    "message": msg,
                    "bundle": {
                        "bundle_id": selected_bundle.bundle_id,
                        "mode": selected_bundle.mode,
                        "version": selected_bundle.model_version,
                        "label": selected_bundle.label,
                    },
                },
                status=200,
            )

        if action == "status":
            job_id = (request.query.get("job_id") or "").strip()
            if not job_id:
                return web.json_response({"status": "error", "message": "Missing job_id in query"}, status=400)
            try:
                status_payload = await maybe_await(svc.get_status(job_id))
            except FileNotFoundError:
                return web.json_response({"status": "error", "message": f"job_id not found: {job_id}"}, status=404)
            except Exception as exc:
                return web.json_response({"status": "error", "message": f"Failed to read status: {exc}"}, status=500)

            return web.json_response({"status": "success", **status_payload}, status=200)

        if action == "training_server_status":
            try:
                status_payload = await maybe_await(svc.get_training_server_connection_status())
            except Exception as exc:
                print(f"Training server connection status failed: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                return web.json_response({"status": "error", "message": f"Failed to check training server connection: {exc}"}, status=500)
            return web.json_response(status_payload, status=200)

        if action == "training_server_detect":
            try:
                detect_payload = await maybe_await(svc.detect_training_server())
            except Exception as exc:
                print(f"Training server detection failed: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                return web.json_response({"status": "error", "message": f"Failed to detect training server: {exc}"}, status=500)
            return web.json_response(detect_payload, status=200)

        if action == "sensor_intervals":
            data = await request.json()
            if not isinstance(data, dict) or not data:
                return web.json_response({"status": "error", "message": "No JSON body received"}, status=400)

            full_history = data.get("fullSensorHistoryData")
            appliance_sensor_history = data.get("applianceSensorHistoryData")
            bundle_id = str(data.get("bundle_id") or "").strip() or None
            bundle_mode = str(data.get("bundle_mode") or "online").strip().lower()

            if not isinstance(full_history, list) or len(full_history) == 0:
                return web.json_response({"status": "error", "message": "fullSensorHistoryData must be a non-empty list"}, status=400)
            if not isinstance(appliance_sensor_history, list) or len(appliance_sensor_history) == 0:
                return web.json_response({"status": "error", "message": "applianceSensorHistoryData must be a non-empty list"}, status=400)

            selected_bundle = None
            if bundle_id:
                selected_bundle = get_bundle_by_id(app_state.model_bundles, bundle_id)
                if selected_bundle is None:
                    return web.json_response({"status": "error", "message": f"Unknown bundle_id: {bundle_id}"}, status=400)
            else:
                selected_bundle = get_latest_bundle_for_mode(app_state.model_bundles, bundle_mode)

            if selected_bundle is None:
                return web.json_response({"status": "error", "message": f"No {bundle_mode} model bundle is available for training"}, status=400)

            try:
                sensor_max_gap_s = app_state.get_sensor_max_gap_s()
                intervals = _build_sensor_supervision_intervals(
                    full_history=full_history,
                    appliance_sensor_history=appliance_sensor_history,
                    inference_dir=selected_bundle.inference_dir,
                    align_grid="start",
                    max_hold_s=sensor_max_gap_s,
                )
            except Exception as exc:
                print(f"Training sensor interval derivation failed: {exc}", flush=True)
                print(traceback.format_exc(), flush=True)
                return web.json_response({"status": "error", "message": f"Failed to derive sensor intervals: {exc}"}, status=500)

            return web.json_response(
                {
                    "status": "success",
                    "intervals": intervals,
                    "count": len(intervals),
                },
                status=200,
            )

        job_id = (request.query.get("job_id") or "").strip()
        if not job_id:
            return web.json_response({"status": "error", "message": "Missing job_id in query"}, status=400)
        try:
            info = await svc.start_send(job_id)
        except FileNotFoundError:
            return web.json_response({"status": "error", "message": f"job_id not found: {job_id}"}, status=404)
        except Exception as exc:
            print(f"Training send failed local_job_id={job_id}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            return web.json_response({"status": "error", "message": f"Failed to start training server job: {exc}"}, status=502)

        return web.json_response(
            {
                "status": "accepted",
                "message": "Training server job started. Poll ?action=status for completion.",
                "job_id": job_id,
                **info,
            },
            status=202,
        )
    except json.JSONDecodeError:
        return web.json_response({"status": "error", "message": "Invalid JSON body"}, status=400)
    except Exception as exc:
        print(f"Training route internal error: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


def register_training_routes(app, ingress_url_base):
    app.router.add_post(ingress_url_base + "training", receive_training_data_handler)
    app.router.add_get(ingress_url_base + "training", receive_training_data_handler)
