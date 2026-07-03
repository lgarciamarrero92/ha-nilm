import aiohttp
from aiohttp import ClientError, web
from datetime import datetime, timezone

import app_state
from ha_client import HistoryQuery, fetch_history_points, points_to_xy_json


async def get_sensors_handler(request):
    headers = {
        "Authorization": f"Bearer {app_state.TOKEN}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{app_state.HA_REST_API_URL}/states", headers=headers) as response:
                response.raise_for_status()
                all_states = await response.json()

            power_units = {"W", "kW"}
            energy_units = {"Wh", "kWh", "MWh", "GWh"}
            power_sensors = []
            for state in all_states:
                entity_id = state.get("entity_id")
                attributes = state.get("attributes", {})
                if not entity_id or not entity_id.startswith("sensor."):
                    continue

                device_class = attributes.get("device_class")
                unit = attributes.get("unit_of_measurement")
                is_power_device_class = device_class == "power"
                is_energy_device_class = device_class == "energy"
                is_power_unit = unit in power_units
                is_energy_unit = unit in energy_units
                if is_energy_device_class or is_energy_unit:
                    continue
                if not (is_power_device_class or is_power_unit):
                    continue

                source = attributes.get("source")
                is_virtual = source == "dl" or entity_id.startswith("sensor.nilm_")
                power_sensors.append({
                    "entity_id": entity_id,
                    "friendly_name": attributes.get("friendly_name", entity_id),
                    "state": state.get("state"),
                    "last_changed": state.get("last_changed"),
                    "device_class": attributes.get("device_class"),
                    "state_class": attributes.get("state_class"),
                    "unit_of_measurement": attributes.get("unit_of_measurement"),
                    "source": source,
                    "is_virtual": is_virtual,
                })

            power_sensors.sort(key=lambda item: item["friendly_name"].lower())
            return web.json_response(power_sensors)
        except aiohttp.ClientError as exc:
            print(f"Error fetching states from HA: {exc}")
            return web.json_response({"status": "error", "message": f"Could not fetch sensors from Home Assistant: {exc}"}, status=500)
        except Exception as exc:
            print(f"Unexpected error in get_sensors_handler: {exc}")
            return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def get_binary_sensors_handler(request):
    headers = {
        "Authorization": f"Bearer {app_state.TOKEN}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{app_state.HA_REST_API_URL}/states", headers=headers) as response:
                response.raise_for_status()
                all_states = await response.json()

            binary_sensors = []
            for state in all_states:
                entity_id = state.get("entity_id")
                attributes = state.get("attributes", {})
                if not entity_id or not entity_id.startswith("binary_sensor."):
                    continue

                source = attributes.get("source")
                is_virtual = source == "dl" or entity_id.startswith("binary_sensor.nilm_")
                binary_sensors.append({
                    "entity_id": entity_id,
                    "friendly_name": attributes.get("friendly_name", entity_id),
                    "state": state.get("state"),
                    "last_changed": state.get("last_changed"),
                    "device_class": attributes.get("device_class"),
                    "source": source,
                    "is_virtual": is_virtual,
                })

            binary_sensors.sort(key=lambda item: item["friendly_name"].lower())
            return web.json_response(binary_sensors)
        except aiohttp.ClientError as exc:
            print(f"Error fetching binary sensors from HA: {exc}")
            return web.json_response({"status": "error", "message": f"Could not fetch binary sensors from Home Assistant: {exc}"}, status=500)
        except Exception as exc:
            print(f"Unexpected error in get_binary_sensors_handler: {exc}")
            return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


async def get_history_handler(request):
    start_time_str_path = request.match_info.get("start_time_str")
    end_time_query = request.query.get("end_time")
    filter_entity_id = request.query.get("filter_entity_id")
    minimal_response_str = request.query.get("minimal_response", "true").lower()
    minimal_response = minimal_response_str != "false"

    if not start_time_str_path:
        return web.json_response({"status": "error", "message": "Missing 'start_time_str' path parameter."}, status=400)

    try:
        if "T" in start_time_str_path:
            start_date_dt = datetime.fromisoformat(start_time_str_path.replace("Z", "+00:00")).astimezone(timezone.utc)
        else:
            start_date_dt = datetime.strptime(start_time_str_path, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return web.json_response(
            {"status": "error", "message": f"Invalid 'start_time_str' path parameter format: {start_time_str_path}. Expected YYYY-MM-DD or ISO UTC datetime."},
            status=400,
        )

    end_date_dt = datetime.now(timezone.utc)
    if end_time_query:
        try:
            end_date_dt = datetime.fromisoformat(end_time_query.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return web.json_response(
                {"status": "error", "message": f"Invalid 'end_time' query parameter format: {end_time_query}. Expected YYYY-MM-DDTHH:MM:SSZ."},
                status=400,
            )

    if start_date_dt >= end_date_dt:
        return web.json_response({"status": "error", "message": "Start date must be before end date."}, status=400)

    span_seconds = (end_date_dt - start_date_dt).total_seconds()
    if span_seconds > (7 * 24 * 60 * 60) + 1:
        return web.json_response(
            {"status": "error", "message": "Date range cannot exceed 7 days."},
            status=400,
        )

    sensor_to_fetch = (filter_entity_id or app_state.current_config.get("main_sensor_id") or "").strip()
    if not sensor_to_fetch:
        return web.json_response(
            {"status": "error", "message": "No mains sensor configured. Please select and save a mains sensor first."},
            status=400,
        )

    try:
        query = HistoryQuery(
            entity_id=sensor_to_fetch,
            start_dt=start_date_dt,
            end_dt=end_date_dt,
            minimal_response=minimal_response,
            max_span_days=7,
        )
        points = await fetch_history_points(app_state.HA_REST_API_URL, app_state.TOKEN, query)
        return web.json_response([points_to_xy_json(points, expand_held_values=True, end_dt=end_date_dt)])
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except RuntimeError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=500)
    except ClientError as exc:
        return web.json_response({"status": "error", "message": f"Could not fetch history from Home Assistant: {exc}"}, status=500)
    except Exception as exc:
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


def register_ha_routes(app, ingress_url_base):
    app.router.add_get("/history/period/{start_time_str}", get_history_handler)
    app.router.add_get(ingress_url_base + "sensors", get_sensors_handler)
    app.router.add_get(ingress_url_base + "binary-sensors", get_binary_sensors_handler)
