from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

from embedding_store import bundle_models_dir, list_saved_models, load_embedding_metadata
from model_registry import ModelBundle, make_model_key
from refquery import RefQueryDisaggregator


HistoryFetcher = Callable[[datetime, datetime], Awaitable[list[tuple[float, float]]]]


class MultiBundleOnlineRuntime:
    def __init__(
        self,
        *,
        bundles: Iterable[ModelBundle],
        models_root: str,
        history_fetcher: Optional[HistoryFetcher],
        num_threads: int = 2,
        max_gap_s: Optional[float] = None,
        top_k=None,
    ):
        self.models_root = models_root
        self.history_fetcher = history_fetcher
        self.num_threads = int(num_threads)
        self.max_gap_s = max_gap_s
        self.top_k = top_k
        self.bundles = [bundle for bundle in bundles if bundle.mode == "online"]
        self.bundle_map = {bundle.bundle_id: bundle for bundle in self.bundles}
        self.runtimes: Dict[str, RefQueryDisaggregator] = {}
        self.enabled_appliances_by_bundle: Dict[str, list[str]] = {}
        self.last_result: Optional[Dict[str, Any]] = None

        saved_models = list_saved_models(models_root)
        available_bundle_ids = {item["bundle_id"] for item in saved_models}
        for item in saved_models:
            metadata = load_embedding_metadata(bundle_models_dir(models_root, item["bundle_id"]), item["appliance_name"]) or {}
            if bool(metadata.get("publish_online")):
                self.enabled_appliances_by_bundle.setdefault(item["bundle_id"], []).append(item["appliance_name"])

        for bundle in self.bundles:
            if bundle.bundle_id not in available_bundle_ids:
                continue
            if not self.enabled_appliances_by_bundle.get(bundle.bundle_id):
                continue
            embeddings_dir = bundle_models_dir(models_root, bundle.bundle_id)
            self.runtimes[bundle.bundle_id] = RefQueryDisaggregator(
                inference_dir=bundle.inference_dir,
                embeddings_dir=embeddings_dir,
                num_threads=self.num_threads,
                history_fetcher=self.history_fetcher,
                max_gap_s=self.max_gap_s,
                top_k=self.top_k,
            )

    async def disaggregate_next(self, total_power: float, now) -> Dict[str, Any]:
        combined: Dict[str, Any] = {
            "timestamp": getattr(now, "timestamp", lambda: float(now))(),
            "window_end_timestamp": getattr(now, "timestamp", lambda: float(now))(),
            "raw_timestamp": getattr(now, "timestamp", lambda: float(now))(),
            "prediction_delay_s": 0.0,
            "pred_idx": None,
            "appliances": {},
        }

        for bundle_id, runtime in self.runtimes.items():
            bundle = self.bundle_map.get(bundle_id)
            if bundle is None:
                continue
            selected_appliances = self.enabled_appliances_by_bundle.get(bundle_id) or []
            if not selected_appliances:
                continue
            result = await runtime.disaggregate_next(total_power, now, appliances=selected_appliances)
            if not result:
                continue

            appliances = result.get("appliances") or {}
            for appliance_name, values in appliances.items():
                model_key = make_model_key(bundle_id, appliance_name)
                combined["appliances"][model_key] = {
                    **values,
                    "appliance_name": appliance_name,
                    "bundle_id": bundle.bundle_id,
                    "bundle_mode": bundle.mode,
                    "bundle_version": bundle.model_version,
                    "timestamp": float(result.get("timestamp", combined["timestamp"])),
                    "window_end_timestamp": float(result.get("window_end_timestamp", combined["window_end_timestamp"])),
                    "raw_timestamp": float(result.get("raw_timestamp", combined["raw_timestamp"])),
                    "prediction_delay_s": float(result.get("prediction_delay_s", 0.0) or 0.0),
                    "pred_idx": result.get("pred_idx"),
                }

        self.last_result = combined
        return combined
