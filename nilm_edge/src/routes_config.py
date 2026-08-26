from aiohttp import web

import app_state
from training_server_url import is_valid_training_server_url, normalize_training_server_url


def _validate_mains(raw_mains):
    if raw_mains is None:
        return []
    if not isinstance(raw_mains, list):
        raise ValueError("Invalid format for 'mains' (must be list).")
    mains = []
    seen = set()
    for item in raw_mains:
        if not isinstance(item, dict):
            raise ValueError("Each mains entry must be an object.")
        sensor_id = str(item.get("sensor_id") or "").strip()
        if not sensor_id:
            raise ValueError("Each mains entry requires 'sensor_id'.")
        if sensor_id in seen:
            raise ValueError("Mains sensor IDs must be unique.")
        seen.add(sensor_id)
        label = str(item.get("label") or "").strip() or sensor_id
        unit = (str(item.get("unit")).strip() if item.get("unit") is not None else None) or None
        mains.append({"sensor_id": sensor_id, "label": label, "unit": unit})
    return mains


def _validate_appliance_mains(raw_assignments, mains=None):
    if raw_assignments is None:
        return {}
    if not isinstance(raw_assignments, dict):
        raise ValueError("Invalid format for 'appliance_mains' (must be object).")
    valid_mains_ids = {
        str(item.get("sensor_id") or "").strip()
        for item in (mains if mains is not None else app_state.get_mains())
        if isinstance(item, dict)
    }
    assignments = {}
    for model_key, sensor_id in raw_assignments.items():
        key = str(model_key or "").strip()
        if not key:
            continue
        assigned_sensor_id = None if sensor_id is None else str(sensor_id).strip()
        assignments[key] = assigned_sensor_id if assigned_sensor_id in valid_mains_ids else None
    return assignments


async def get_config_handler(request):
    training_server_state = await app_state.resolve_training_server_url_state()
    return web.json_response({
        **app_state.current_config,
        **training_server_state,
    })


async def post_config_handler(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        update_main_sensor_id = "main_sensor_id" in data
        update_main_sensor_unit = "main_sensor_unit" in data
        update_mains = "mains" in data
        update_appliance_mains = "appliance_mains" in data
        update_training_server_url = "training_server_url" in data
        if not update_main_sensor_id and not update_main_sensor_unit and not update_mains and not update_appliance_mains and not update_training_server_url:
            raise ValueError("Nothing to update. Provide 'main_sensor_id', 'main_sensor_unit', 'mains', 'appliance_mains' and/or 'training_server_url'.")

        new_main_sensor_id = data.get("main_sensor_id")
        new_main_sensor_unit = data.get("main_sensor_unit")
        new_mains = _validate_mains(data.get("mains")) if update_mains else None
        assignment_mains = new_mains if update_mains else None
        new_appliance_mains = _validate_appliance_mains(data.get("appliance_mains"), assignment_mains) if update_appliance_mains else None
        new_training_server_url = data.get("training_server_url")

        if update_main_sensor_id and new_main_sensor_id is not None and not isinstance(new_main_sensor_id, str):
            raise ValueError("Invalid format for 'main_sensor_id' (must be string).")
        if update_main_sensor_unit and new_main_sensor_unit is not None and not isinstance(new_main_sensor_unit, str):
            raise ValueError("Invalid format for 'main_sensor_unit' (must be string).")
        if update_training_server_url and new_training_server_url is not None and not isinstance(new_training_server_url, str):
            raise ValueError("Invalid format for 'training_server_url' (must be string).")
        if update_training_server_url and isinstance(new_training_server_url, str) and new_training_server_url.strip():
            normalized_training_server_url = normalize_training_server_url(new_training_server_url)
            if not is_valid_training_server_url(normalized_training_server_url):
                raise ValueError("Invalid 'training_server_url'. Use a full host or URL such as http://trainer.local:8080/train.")

        app_state.save_config(
            main_sensor_id=new_main_sensor_id,
            main_sensor_unit=new_main_sensor_unit,
            mains=new_mains,
            appliance_mains=new_appliance_mains,
            training_server_url=new_training_server_url,
            update_main_sensor_id=update_main_sensor_id,
            update_main_sensor_unit=update_main_sensor_unit,
            update_mains=update_mains,
            update_appliance_mains=update_appliance_mains,
            update_training_server_url=update_training_server_url,
        )
        if update_main_sensor_id or update_mains or update_appliance_mains:
            app_state.reload_algorithm_config()

        return web.json_response({"status": "success", "message": "Configuration updated successfully."})
    except ValueError as exc:
        return web.json_response({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        print(f"Error handling POST /config: {exc}")
        return web.json_response({"status": "error", "message": f"Internal server error: {exc}"}, status=500)


def register_config_routes(app, ingress_url_base):
    app.router.add_get(ingress_url_base + "config", get_config_handler)
    app.router.add_post(ingress_url_base + "config", post_config_handler)
