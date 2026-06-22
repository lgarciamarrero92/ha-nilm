import asyncio
import json
import os
import signal
import time
from datetime import datetime, timezone

import aiohttp
from aiohttp import web
import websockets

import app_state
from embedding_store import save_embedding_npy
from energy_accumulator import EnergyAccumulator
from power_units import normalize_power_to_watts
from routes_config import register_config_routes
from routes_ha import register_ha_routes
from routes_models import register_model_routes
from routes_training import register_training_routes
from training_server_service import TrainingServerServiceManager


running = True
energy_accumulator = EnergyAccumulator("/data/nilm_energy.json")


def shutdown_handler(sig, frame):
    global running
    print("Received shutdown signal")
    running = False


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


def _slug(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_").lower()


async def publish_disaggregation_dl(
    total_power: float,
    dl_result,
    timestamp: datetime,
    session: aiohttp.ClientSession,
    duration=None,
):
    if not dl_result or not isinstance(dl_result, dict):
        return

    appliances = dl_result.get("appliances") or {}
    if not isinstance(appliances, dict):
        return

    prediction_target_dt = datetime.fromtimestamp(
        float(dl_result.get("timestamp", timestamp.timestamp())),
        tz=timezone.utc,
    )
    window_end_dt = datetime.fromtimestamp(
        float(dl_result.get("window_end_timestamp", timestamp.timestamp())),
        tz=timezone.utc,
    )
    raw_sample_dt = datetime.fromtimestamp(
        float(dl_result.get("raw_timestamp", timestamp.timestamp())),
        tz=timezone.utc,
    )
    prediction_delay_s = float(dl_result.get("prediction_delay_s", 0.0) or 0.0)
    pred_idx = dl_result.get("pred_idx")

    headers = {
        "Authorization": f"Bearer {app_state.TOKEN}",
        "Content-Type": "application/json",
    }

    async def post_state(entity_id: str, payload: dict) -> None:
        try:
            async with session.post(
                f"{app_state.HA_REST_API_URL}/states/{entity_id}",
                headers=headers,
                json=payload,
            ) as response:
                if response.status not in [200, 201]:
                    response_text = await response.text()
                    print(f"Error updating {entity_id} via REST API: {response.status} - {response_text}")
        except Exception as exc:
            print(f"Exception during REST API call for {entity_id}: {exc}")

    publish_tasks = []

    for appliance_name, values in appliances.items():
        try:
            power = float(values.get("power"))
            onoff = float(values.get("onoff"))
        except Exception:
            continue
        try:
            onoff_threshold = float(values.get("onoff_threshold", 0.5))
        except Exception:
            onoff_threshold = 0.5

        display_name = str(values.get("appliance_name") or appliance_name)
        bundle_id = str(values.get("bundle_id") or "").strip()
        bundle_label = f" [{bundle_id}]" if bundle_id else ""
        slug = _slug(f"{display_name}_{bundle_id}" if bundle_id else display_name)
        appliance_prediction_target_dt = datetime.fromtimestamp(
            float(values.get("timestamp", prediction_target_dt.timestamp())),
            tz=timezone.utc,
        )
        appliance_window_end_dt = datetime.fromtimestamp(
            float(values.get("window_end_timestamp", window_end_dt.timestamp())),
            tz=timezone.utc,
        )
        appliance_raw_sample_dt = datetime.fromtimestamp(
            float(values.get("raw_timestamp", raw_sample_dt.timestamp())),
            tz=timezone.utc,
        )
        appliance_prediction_delay_s = float(values.get("prediction_delay_s", prediction_delay_s) or 0.0)
        appliance_pred_idx = values.get("pred_idx", pred_idx)
        power_entity_id = f"sensor.nilm_{slug}_power"
        # Keep this distinct from the common user-created integral helper name
        # (sensor.nilm_<appliance>_energy) so upgrades cannot overwrite it.
        energy_entity_id = f"sensor.nilm_{slug}_energy_consumed"
        energy_update = energy_accumulator.update(
            appliance_key=energy_entity_id,
            power_w=power,
            timestamp_s=timestamp.timestamp(),
        )
        power_data = {
            "state": round(power, 1),
            "attributes": {
                "unit_of_measurement": "W",
                "device_class": "power",
                "state_class": "measurement",
                "friendly_name": f"NILM {display_name.replace('_', ' ').title()} Power{bundle_label}",
                "last_updated": timestamp.isoformat(),
                "prediction_target_time": appliance_prediction_target_dt.isoformat(),
                "window_end_time": appliance_window_end_dt.isoformat(),
                "raw_sample_time": appliance_raw_sample_dt.isoformat(),
                "prediction_delay_s": round(appliance_prediction_delay_s, 3),
                "pred_idx": appliance_pred_idx,
                "bundle_id": bundle_id or None,
                "icon": "mdi:power-socket-eu",
                "source": "dl",
                "onoff_score": round(onoff, 4),
                "onoff_threshold": round(onoff_threshold, 4),
            },
        }
        publish_tasks.append(post_state(power_entity_id, power_data))

        energy_data = {
            "state": round(energy_update.total_kwh, 6),
            "attributes": {
                "unit_of_measurement": "kWh",
                "device_class": "energy",
                "state_class": "total",
                "friendly_name": f"NILM {display_name.replace('_', ' ').title()} Energy Consumed{bundle_label}",
                "last_updated": timestamp.isoformat(),
                "source": "dl",
                "integration_method": "trapezoidal",
                "integration_gap_s": round(energy_update.integration_gap_s, 3),
                "skipped_stale_gap": energy_update.skipped_stale_gap,
                "icon": "mdi:lightning-bolt",
            },
        }
        publish_tasks.append(post_state(energy_entity_id, energy_data))

        is_on = onoff >= onoff_threshold
        on_entity_id = f"binary_sensor.nilm_{slug}_on"
        on_data = {
            "state": "on" if is_on else "off",
            "attributes": {
                "friendly_name": f"NILM {display_name.replace('_', ' ').title()} On/Off{bundle_label}",
                "last_updated": timestamp.isoformat(),
                "prediction_target_time": appliance_prediction_target_dt.isoformat(),
                "window_end_time": appliance_window_end_dt.isoformat(),
                "raw_sample_time": appliance_raw_sample_dt.isoformat(),
                "prediction_delay_s": round(appliance_prediction_delay_s, 3),
                "pred_idx": appliance_pred_idx,
                "bundle_id": bundle_id or None,
                "icon": "mdi:toggle-switch" if is_on else "mdi:toggle-switch-off",
                "source": "dl",
                "onoff_score": round(onoff, 4),
                "onoff_threshold": round(onoff_threshold, 4),
            },
        }
        publish_tasks.append(post_state(on_entity_id, on_data))

    if duration is not None:
        duration_entity_id = "sensor.nilm_disaggregation_duration"
        duration_data = {
            "state": round(float(duration), 3),
            "attributes": {
                "unit_of_measurement": "s",
                "device_class": "duration",
                "state_class": "measurement",
                "friendly_name": "NILM Disaggregation Duration",
                "last_updated": timestamp.isoformat(),
                "prediction_target_time": prediction_target_dt.isoformat(),
                "window_end_time": window_end_dt.isoformat(),
                "raw_sample_time": raw_sample_dt.isoformat(),
                "prediction_delay_s": round(prediction_delay_s, 3),
                "pred_idx": pred_idx,
                "icon": "mdi:timer-outline",
                "source": "dl",
            },
        }
        publish_tasks.append(post_state(duration_entity_id, duration_data))

    if publish_tasks:
        await asyncio.gather(*publish_tasks, return_exceptions=True)


async def retry_websocket_connection(url, max_retries=10, initial_delay=1):
    for attempt in range(max_retries):
        try:
            print(f"Attempting WebSocket connection to {url} (attempt {attempt + 1}/{max_retries})...")
            websocket = await websockets.connect(url)
            print("WebSocket connection established.")
            return websocket
        except Exception as exc:
            delay = initial_delay * (2 ** attempt)
            print(f"WebSocket connection failed: {exc}. Retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)
    raise ConnectionRefusedError(f"Failed to establish WebSocket connection to {url} after {max_retries} attempts.")


def build_web_app():
    app = web.Application(client_max_size=50 * 1024**2)
    app["training_server_manager"] = TrainingServerServiceManager(
        jobs_dir="/data/training_jobs",
        models_root=app_state.MODELS_ROOT,
        training_server_url=app_state.get_training_server_url(),
        training_server_api_key=app_state.get_training_server_api_key(),
        save_embedding_npy_fn=save_embedding_npy,
        reload_algorithm_fn=app_state.reload_algorithm_config,
    )

    app.router.add_get(
        app_state.INGRESS_URL_BASE,
        lambda request: web.FileResponse(os.path.join("/app/www", "index.html")),
    )
    if app_state.INGRESS_URL_BASE != "/":
        app.router.add_get(
            app_state.INGRESS_URL_BASE.rstrip("/"),
            lambda request: web.FileResponse(os.path.join("/app/www", "index.html")),
        )

    app.router.add_static(app_state.INGRESS_URL_BASE + "components/", path="/app/www/components")
    app.router.add_static(app_state.INGRESS_URL_BASE + "js/", path="/app/www/js")
    app.router.add_static(app_state.INGRESS_URL_BASE + "vendor/", path="/app/www/vendor")
    app.router.add_get(
        app_state.INGRESS_URL_BASE + "icon.png",
        lambda request: web.FileResponse(os.path.join("/app/www", "icon.png")),
    )

    register_config_routes(app, app_state.INGRESS_URL_BASE)
    register_model_routes(app, app_state.INGRESS_URL_BASE)
    register_ha_routes(app, app_state.INGRESS_URL_BASE)
    register_training_routes(app, app_state.INGRESS_URL_BASE)
    return app


async def run_live_loop(session: aiohttp.ClientSession):
    websocket = None
    backoff = 1
    subscribed_sensor = None

    while running:
        try:
            sensor_to_monitor = app_state.current_config.get("main_sensor_id")
            websocket = await retry_websocket_connection(app_state.HA_WS_URL)

            initial_auth_reply = json.loads(await websocket.recv())
            print(f"Received initial server message: {initial_auth_reply}")
            if initial_auth_reply.get("type") != "auth_required":
                raise RuntimeError(
                    f"Expected 'auth_required' from HA, but got: {initial_auth_reply.get('type')}. Full reply: {initial_auth_reply}"
                )

            await websocket.send(json.dumps({"type": "auth", "access_token": app_state.TOKEN}))
            auth_result = json.loads(await websocket.recv())
            print(f"Received authentication result: {auth_result}")

            if auth_result.get("type") == "auth_ok":
                print("Home Assistant WebSocket authentication successful!")
            elif auth_result.get("type") == "auth_invalid":
                raise RuntimeError(f"HA WebSocket authentication failed: {auth_result.get('message', 'Invalid token provided.')}")
            else:
                raise RuntimeError(
                    f"Unexpected WebSocket authentication response type: {auth_result.get('type')}. Full reply: {auth_result}"
                )

            await websocket.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
            if sensor_to_monitor:
                print(f"Listening to {sensor_to_monitor} via HA WebSocket...")
            else:
                print("No mains sensor configured yet. Waiting for the user to select one before running NILM.")
            subscribed_sensor = sensor_to_monitor
            backoff = 1

            while running:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=60)
                except asyncio.TimeoutError:
                    print("No data received in 60s.")
                    if app_state.refquery_instance and app_state.current_config.get("main_sensor_id"):
                        now = datetime.now(timezone.utc)
                        start_time = time.perf_counter()
                        dl_disagg = await app_state.refquery_instance.disaggregate_next(0.0, now)
                        dl_dur = time.perf_counter() - start_time
                        try:
                            await publish_disaggregation_dl(0.0, dl_disagg, now, session, dl_dur)
                        except Exception as pub_err:
                            print(f"Publish error during idle tick: {pub_err}")
                    else:
                        print("NILM instance not available, skipping idle tick.")
                    continue
                except websockets.exceptions.ConnectionClosedOK:
                    print("WebSocket connection closed gracefully.")
                    break

                try:
                    sensor_to_monitor = app_state.current_config.get("main_sensor_id")
                    if subscribed_sensor != sensor_to_monitor:
                        if sensor_to_monitor:
                            print(f"Detected mains sensor change. Now filtering events for {sensor_to_monitor}.")
                        else:
                            print("Main sensor selection cleared. NILM will stay idle until a mains sensor is saved.")
                        subscribed_sensor = sensor_to_monitor

                    event = json.loads(msg)
                    if not sensor_to_monitor:
                        continue
                    new_state = event.get("event", {}).get("data", {}).get("new_state")
                    if not new_state or new_state.get("entity_id") != sensor_to_monitor:
                        continue

                    sensor_unit = (
                        new_state.get("attributes", {}).get("unit_of_measurement")
                        or app_state.current_config.get("main_sensor_unit")
                    )
                    if sensor_unit and sensor_unit != app_state.current_config.get("main_sensor_unit"):
                        app_state.current_config["main_sensor_unit"] = str(sensor_unit).strip()

                    total_power = normalize_power_to_watts(new_state["state"], sensor_unit)
                    now = datetime.now(timezone.utc)

                    if app_state.refquery_instance:
                        start_time = time.perf_counter()
                        dl_disagg = await app_state.refquery_instance.disaggregate_next(total_power, now)
                        dl_dur = time.perf_counter() - start_time
                        await publish_disaggregation_dl(total_power, dl_disagg, now, session, dl_dur)
                    else:
                        print("NILM instance not available, skipping disaggregation.")
                except Exception as exc:
                    print(f"Error during loop: {exc}")

        except Exception as err:
            print(f"WebSocket connection error: {err}")
            sleep_s = backoff
            backoff = min(backoff * 2, 60)
            print(f"Reconnecting in {sleep_s}s...")
            await asyncio.sleep(sleep_s)
            continue
        finally:
            if websocket is not None and not websocket.closed:
                print("Closing WebSocket connection gracefully...")
                try:
                    await websocket.close()
                except Exception:
                    pass


async def main():
    print("Main application starting...")
    print(f"Configured training server URL override: {app_state.get_configured_training_server_url() or '(auto)'}", flush=True)
    if not app_state.TOKEN:
        print("ERROR: SUPERVISOR_TOKEN environment variable is not set!")
        return

    app_state.load_config()
    app_state.reload_algorithm_config()

    app = build_web_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8099)
    await site.start()
    print("Web server started on port 8099 for Ingress.")

    timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            await run_live_loop(session)
        finally:
            print("Shutting down NILM service.")
            await runner.cleanup()
            print("Web server stopped.")


if __name__ == "__main__":
    os.makedirs("/data", exist_ok=True)
    asyncio.run(main())
