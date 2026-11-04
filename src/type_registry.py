"""
Per-tenant/namespace entity-type registry.

EpochDB still auto-discovers types from stored triples (is_a / type / …).
Apps may also predefine or create types via this registry; those are first-class
and merged into /discovered_types and /entity_types responses.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("epochdb_type_registry")

_lock = threading.RLock()

TYPE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _slug(value: str) -> str:
    s = (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s[:64]


def normalize_type_id(raw: str) -> str:
    return _slug(raw)


def validate_type_id(type_id: str) -> str:
    tid = normalize_type_id(type_id)
    if not tid or not TYPE_ID_RE.match(tid):
        raise ValueError(
            f"Invalid type id '{type_id}': use lowercase snake_case "
            "(letter first, max 64 chars)"
        )
    return tid


def _storage_root() -> Path:
    # Coordinator may not mount STORAGE_DIR — allow a dedicated registry path.
    override = (os.getenv("TYPE_REGISTRY_DIR") or "").strip()
    if override:
        return Path(override)
    return Path(os.getenv("STORAGE_DIR", "/data"))


def _registry_path(tenant: Optional[str], namespace: Optional[str]) -> Path:
    t = (tenant or "_default").strip() or "_default"
    n = (namespace or "_default").strip() or "_default"
    # Coordinator has no tenant data dirs by default — keep a shared registry tree.
    base = _storage_root() / "type_registry" / t / n
    return base / "entity_types.json"


def _empty_doc() -> Dict[str, Any]:
    return {"version": 1, "types": {}, "updated_at": time.time()}


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_doc()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_doc()
        data.setdefault("types", {})
        return data
    except Exception as e:
        logger.warning("Failed reading type registry %s: %s", path, e)
        return _empty_doc()


def _write(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    doc = dict(doc)
    doc["updated_at"] = time.time()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def list_registered(
    tenant: Optional[str] = None, namespace: Optional[str] = None
) -> List[Dict[str, Any]]:
    path = _registry_path(tenant, namespace)
    with _lock:
        doc = _read(path)
    types = doc.get("types") or {}
    out = []
    for tid, meta in types.items():
        if not isinstance(meta, dict):
            meta = {"id": tid, "label": tid}
        item = {
            "id": tid,
            "label": meta.get("label") or tid,
            "description": meta.get("description") or "",
            "source": meta.get("source") or "registered",
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
        }
        out.append(item)
    out.sort(key=lambda x: x["id"])
    return out


def get_registered_ids(
    tenant: Optional[str] = None, namespace: Optional[str] = None
) -> List[str]:
    return [t["id"] for t in list_registered(tenant, namespace)]


def upsert_type(
    type_id: str,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    source: str = "app",
    tenant: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    tid = validate_type_id(type_id)
    path = _registry_path(tenant, namespace)
    now = time.time()
    with _lock:
        doc = _read(path)
        types = doc.setdefault("types", {})
        prev = types.get(tid) if isinstance(types.get(tid), dict) else {}
        entry = {
            "id": tid,
            "label": (label or prev.get("label") or tid.replace("_", " ").title()),
            "description": description if description is not None else prev.get("description", ""),
            "source": source or prev.get("source") or "app",
            "created_at": prev.get("created_at") or now,
            "updated_at": now,
        }
        types[tid] = entry
        _write(path, doc)
    return entry


def seed_types(
    entries: List[Dict[str, Any]],
    *,
    replace: bool = False,
    source: str = "seed",
    tenant: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    path = _registry_path(tenant, namespace)
    now = time.time()
    created = 0
    updated = 0
    with _lock:
        doc = _empty_doc() if replace else _read(path)
        types = doc.setdefault("types", {})
        if replace:
            types.clear()
        for raw in entries or []:
            if isinstance(raw, str):
                tid = validate_type_id(raw)
                label = tid.replace("_", " ").title()
                description = ""
            elif isinstance(raw, dict):
                tid = validate_type_id(raw.get("id") or raw.get("type") or "")
                label = raw.get("label") or tid.replace("_", " ").title()
                description = raw.get("description") or ""
            else:
                continue
            prev = types.get(tid) if isinstance(types.get(tid), dict) else None
            if prev:
                updated += 1
                created_at = prev.get("created_at") or now
            else:
                created += 1
                created_at = now
            types[tid] = {
                "id": tid,
                "label": label,
                "description": description,
                "source": source,
                "created_at": created_at,
                "updated_at": now,
            }
        _write(path, doc)
    return {
        "status": "ok",
        "created": created,
        "updated": updated,
        "total": len(get_registered_ids(tenant, namespace)),
        "types": list_registered(tenant, namespace),
    }


def delete_type(
    type_id: str,
    *,
    tenant: Optional[str] = None,
    namespace: Optional[str] = None,
) -> bool:
    tid = normalize_type_id(type_id)
    path = _registry_path(tenant, namespace)
    with _lock:
        doc = _read(path)
        types = doc.get("types") or {}
        if tid not in types:
            return False
        del types[tid]
        _write(path, doc)
    return True
