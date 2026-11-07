from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional

from app.heads.refquery_dense_v1 import Head as RefQueryDenseV1


HEAD_IMPLEMENTATIONS = {
    "refquery_dense_v1": RefQueryDenseV1,
}


@dataclass(frozen=True)
class ServerModelBundle:
    bundle_id: str
    mode: str
    model_version: int
    saved_at: str
    display_name: Optional[str]
    is_default_for_training: bool
    bundle_dir: str
    load_head: Callable

    @property
    def label(self) -> str:
        return self.display_name or f"{self.mode.title()} v{self.model_version}"


def _load_bundle(manifest_path: str) -> ServerModelBundle:
    with open(manifest_path, "r", encoding="utf-8") as file_handle:
        raw = json.load(file_handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid bundle manifest: {manifest_path}")

    bundle_id = str(raw.get("bundle_id") or "").strip()
    if not bundle_id:
        raise ValueError(f"bundle_id is required in {manifest_path}")

    mode = str(raw.get("mode") or "online").strip().lower()
    bundle_dir = os.path.dirname(manifest_path)
    head_implementation = str(raw.get("head_implementation") or "").strip()
    if head_implementation not in HEAD_IMPLEMENTATIONS:
        raise ValueError(f"Unknown head_implementation {head_implementation!r} for server bundle {bundle_id}.")
    weights_path = os.path.join(bundle_dir, "head.npz")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing head.npz for server bundle {bundle_id}: {weights_path}")

    return ServerModelBundle(
        bundle_id=bundle_id,
        mode=mode,
        model_version=int(raw.get("model_version", raw.get("version", 1))),
        saved_at=str(raw.get("saved_at") or ""),
        display_name=str(raw.get("display_name") or "").strip() or None,
        is_default_for_training=bool(raw.get("is_default_for_training", False)),
        bundle_dir=bundle_dir,
        load_head=partial(HEAD_IMPLEMENTATIONS[head_implementation], weights_path),
    )


def discover_server_bundles(bundles_root: str) -> List[ServerModelBundle]:
    bundles: Dict[str, ServerModelBundle] = {}
    if not os.path.isdir(bundles_root):
        return []

    for entry in os.listdir(bundles_root):
        manifest_path = os.path.join(bundles_root, entry, "bundle_manifest.json")
        if not os.path.exists(manifest_path):
            continue
        bundle = _load_bundle(manifest_path)
        bundles[bundle.bundle_id] = bundle

    return sorted(
        bundles.values(),
        key=lambda item: (int(item.model_version), item.saved_at, item.bundle_id),
        reverse=True,
    )


def get_bundle_by_id(bundles: Iterable[ServerModelBundle], bundle_id: str) -> Optional[ServerModelBundle]:
    wanted = str(bundle_id or "").strip()
    for bundle in bundles:
        if bundle.bundle_id == wanted:
            return bundle
    return None


def get_latest_bundle_for_mode(bundles: Iterable[ServerModelBundle], mode: str) -> Optional[ServerModelBundle]:
    normalized_mode = str(mode or "").strip().lower()
    candidates = [bundle for bundle in bundles if bundle.mode == normalized_mode]
    if not candidates:
        return None
    explicit_defaults = [bundle for bundle in candidates if bundle.is_default_for_training]
    if explicit_defaults:
        return explicit_defaults[0]
    return candidates[0]
