import inspect
import json
import os
from typing import Any, Dict, Optional

import aiohttp

from embedding_store import migrate_legacy_models, rename_bundle_models_dir
from ha_client import HistoryQuery, fetch_history_points
from model_registry import discover_model_bundles, get_latest_bundle_for_mode
from online_runtime import MultiBundleOnlineRuntime
from power_units import normalize_power_to_watts
from runtime_settings import DEFAULT_SENSOR_MAX_GAP_S, clamp_sensor_max_gap_s
from supervisor_addons import discover_training_server_addon
from training_server_url import normalize_training_server_url, uses_homeassistant_gateway


TRAINING_SERVER_API_KEY = os.getenv("TRAINING_SERVER_API_KEY", "").strip() or None
MODELS_ROOT = "/data/models"
LEGACY_EMBEDDINGS_DIR = "/data/embeddings"
INFERENCE_ROOT = "/app/inference"
CONFIG_FILE_PATH = "/data/config.json"
OPTIONS_FILE_PATH = "/data/options.json"
SUPERVISOR_API_URL = os.getenv("SUPERVISOR_API_URL", "http://supervisor")
DEFAULT_BATCH_SIZE = 1024

HA_WS_URL = os.getenv("HA_WS_URL", "ws://supervisor/core/websocket")
HA_REST_API_URL = os.getenv("HA_REST_API_URL", "http://supervisor/core/api")
TOKEN = os.getenv("SUPERVISOR_TOKEN")

INGRESS_URL_BASE = os.getenv("SUPERVISOR_INGRESS_URL", "/")
if not INGRESS_URL_BASE.endswith("/"):
    INGRESS_URL_BASE += "/"

current_config = {
    "main_sensor_id": (os.getenv("MAIN_SENSOR", "").strip() or None),
    "main_sensor_unit": None,
    "training_server_url": None,
}

refquery_instance = None
model_bundles = []


async def maybe_await(value):
    return await value if inspect.isawaitable(value) else value


async def resolve_sensor_unit(entity_id: Optional[str]) -> Optional[str]:
    sensor = str(entity_id or "").strip()
    if not sensor:
        return None

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{HA_REST_API_URL}/states/{sensor}", headers=headers) as response:
                response.raise_for_status()
                payload = await response.json()
        unit = str(payload.get("attributes", {}).get("unit_of_measurement") or "").strip() or None
        if unit:
            current_config["main_sensor_unit"] = unit
        return unit
    except Exception as exc:
        print(f"Warning: could not resolve unit for {sensor}: {exc}")
        return None


def get_training_server_url() -> str:
    local_url = str(current_config.get("training_server_url") or "").strip()
    if local_url:
        return normalize_training_server_url(local_url)
    return ""


def get_configured_training_server_url() -> str:
    return str(current_config.get("training_server_url") or "").strip()


def get_training_server_api_key() -> Optional[str]:
    return TRAINING_SERVER_API_KEY


def _clamp_batch_size(value) -> int:
    try:
        return max(32, min(8192, int(value)))
    except (TypeError, ValueError):
        return DEFAULT_BATCH_SIZE


def _load_options() -> Dict[str, Any]:
    if not os.path.exists(OPTIONS_FILE_PATH):
        return {}

    try:
        with open(OPTIONS_FILE_PATH, "r", encoding="utf-8") as file_handle:
            loaded_options = json.load(file_handle)
        if not isinstance(loaded_options, dict):
            return {}
        return loaded_options
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading add-on options from {OPTIONS_FILE_PATH}: {exc}. Using default options.")
        return {}


def get_batch_size() -> int:
    return _clamp_batch_size(_load_options().get("batch_size"))


def get_sensor_max_gap_s() -> float:
    env_value = os.getenv("SENSOR_MAX_GAP_S")
    if env_value:
        return clamp_sensor_max_gap_s(env_value)
    options = _load_options()
    if "sensor_max_gap_s" not in options:
        return DEFAULT_SENSOR_MAX_GAP_S
    return clamp_sensor_max_gap_s(options.get("sensor_max_gap_s"))


async def history_fetcher(start_dt, end_dt):
    sensor_id = current_config.get("main_sensor_id")
    if not sensor_id:
        return []

    sensor_unit = current_config.get("main_sensor_unit") or await resolve_sensor_unit(sensor_id)

    query = HistoryQuery(
        entity_id=sensor_id,
        start_dt=start_dt,
        end_dt=end_dt,
        minimal_response=True,
        max_span_days=7,
    )
    raw_points = await fetch_history_points(HA_REST_API_URL, TOKEN, query)
    try:
        return [(float(ts), normalize_power_to_watts(value, sensor_unit)) for ts, value in raw_points]
    except Exception:
        return raw_points


def _is_direct_training_server_url(url: str) -> bool:
    normalized = normalize_training_server_url(url)
    if not normalized:
        return False
    return not uses_homeassistant_gateway(normalized)


def _append_training_server_option(options_list, seen_urls, *, option_id: str, label: str, url: str, description: str = ""):
    normalized = normalize_training_server_url(url)
    if not normalized or normalized in seen_urls:
        return
    seen_urls.add(normalized)
    options_list.append({
        "id": option_id,
        "label": label,
        "url": normalized,
        "description": description,
    })


async def resolve_training_server_url_state() -> Dict[str, Any]:
    available_training_servers = []
    seen_urls = set()
    configured_url = get_configured_training_server_url()
    normalized_configured_url = normalize_training_server_url(configured_url)
    autodetect = await discover_training_server_addon(SUPERVISOR_API_URL, TOKEN)
    autodetected_url = normalize_training_server_url(str(autodetect.get("training_server_url") or "").strip())

    if autodetect.get("ok") and autodetect.get("training_server_url"):
        hostname = autodetect.get("hostname") or "internal app"
        _append_training_server_option(
            available_training_servers,
            seen_urls,
            option_id="internal_addon",
            label="Internal App",
            url=autodetect["training_server_url"],
            description=f"Detected internal app hostname: {hostname}",
        )

    configured_matches_autodetect = bool(
        normalized_configured_url
        and autodetected_url
        and normalized_configured_url == autodetected_url
    )

    if normalized_configured_url and _is_direct_training_server_url(normalized_configured_url) and not configured_matches_autodetect:
        _append_training_server_option(
            available_training_servers,
            seen_urls,
            option_id="external_custom",
            label="External Server",
            url=normalized_configured_url,
            description="Saved external training server URL.",
        )
    elif normalized_configured_url:
        _append_training_server_option(
            available_training_servers,
            seen_urls,
            option_id="saved_server",
            label="Saved Server",
            url=normalized_configured_url,
            description="Saved training server selection.",
        )

    effective_training_server_url = ""
    training_server_url_source = "missing"
    if configured_matches_autodetect:
        effective_training_server_url = autodetected_url
        training_server_url_source = "autodetect"
    elif normalized_configured_url:
        effective_training_server_url = normalized_configured_url
        training_server_url_source = "external_custom" if _is_direct_training_server_url(normalized_configured_url) else "saved_config"
    elif autodetect.get("ok") and autodetect.get("training_server_url"):
        effective_training_server_url = autodetected_url
        training_server_url_source = "autodetect"

    return {
        "configured_training_server_url": configured_url,
        "effective_training_server_url": effective_training_server_url,
        "training_server_url_source": training_server_url_source,
        "available_training_servers": available_training_servers,
        "autodetect": autodetect,
    }


def load_config():
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"No config file found at {CONFIG_FILE_PATH}. Using default values.")
        return

    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as file_handle:
            loaded_config = json.load(file_handle)
        loaded_sensor_id = loaded_config.get("main_sensor_id", current_config["main_sensor_id"])
        loaded_sensor_unit = loaded_config.get("main_sensor_unit", current_config["main_sensor_unit"])
        loaded_training_server_url = loaded_config.get("training_server_url", current_config["training_server_url"])
        current_config["main_sensor_id"] = (str(loaded_sensor_id).strip() if loaded_sensor_id is not None else None) or None
        current_config["main_sensor_unit"] = (str(loaded_sensor_unit).strip() if loaded_sensor_unit is not None else None) or None
        current_config["training_server_url"] = (
            normalize_training_server_url(str(loaded_training_server_url).strip())
            if loaded_training_server_url
            else None
        )
        print(f"Configuration loaded from {CONFIG_FILE_PATH}")
    except json.JSONDecodeError as exc:
        print(f"Error decoding config.json: {exc}. Using current in-memory values.")
    except Exception as exc:
        print(f"Error reading config.json: {exc}. Using current in-memory values.")


def save_config(
    *,
    main_sensor_id=None,
    main_sensor_unit=None,
    training_server_url=None,
    update_main_sensor_id=False,
    update_main_sensor_unit=False,
    update_training_server_url=False,
):
    if update_main_sensor_id:
        current_config["main_sensor_id"] = (str(main_sensor_id).strip() if main_sensor_id is not None else None) or None
    if update_main_sensor_unit:
        current_config["main_sensor_unit"] = (str(main_sensor_unit).strip() if main_sensor_unit is not None else None) or None
    if update_training_server_url:
        current_config["training_server_url"] = (
            normalize_training_server_url(str(training_server_url).strip())
            if training_server_url is not None and str(training_server_url).strip()
            else None
        )
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(current_config, file_handle, indent=2)
        print(f"Configuration saved to {CONFIG_FILE_PATH}")
    except Exception as exc:
        print(f"Error saving configuration to {CONFIG_FILE_PATH}: {exc}")


def reload_algorithm_config():
    global refquery_instance
    global model_bundles

    try:
        renamed = rename_bundle_models_dir(MODELS_ROOT, "nilm_online_v1", "online_v1")
        if renamed:
            print("Renamed saved models bundle 'nilm_online_v1' to 'online_v1'.")

        model_bundles = discover_model_bundles(INFERENCE_ROOT)
        default_online_bundle = get_latest_bundle_for_mode(model_bundles, "online")
        migrated = migrate_legacy_models(
            legacy_embeddings_dir=LEGACY_EMBEDDINGS_DIR,
            models_root=MODELS_ROOT,
            default_bundle_id=default_online_bundle.bundle_id if default_online_bundle else None,
        )
        if migrated:
            print(f"Migrated {migrated} legacy model files into bundle-aware storage.")

        refquery_instance = MultiBundleOnlineRuntime(
            bundles=model_bundles,
            models_root=MODELS_ROOT,
            num_threads=2,
            history_fetcher=history_fetcher,
            max_gap_s=get_sensor_max_gap_s(),
            top_k=None,
        )
        print("Algorithm configuration reloaded successfully.")
    except Exception as exc:
        print(f"ERROR: Failed to initialize RefQuery disaggregator: {exc}")
        refquery_instance = None
