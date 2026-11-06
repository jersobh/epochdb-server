"""
Auto-discover entity types from stored triples (open vocabulary).

Uses engine.get_discovered_types when available; otherwise scans the hot tier
for type-defining predicates (is_a, instance_of, …). Does not invent types from
job titles or colon-prefixed labels.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

try:
    from ontology import TYPE_DEFINING_PREDICATES, normalize_type
except ImportError:
    from src.ontology import TYPE_DEFINING_PREDICATES, normalize_type

logger = logging.getLogger("epochdb_type_discovery")

_MAX_ATOMS = 8000


def _pred_key(p: Any) -> str:
    return str(p or "").strip().lower().replace(" ", "_")


def _scan_hot_tier(engine: Any) -> Dict[str, Any]:
    types: Set[str] = set()
    predicates: Set[str] = set()
    entity_types: Dict[str, Set[str]] = {}

    hot = getattr(engine, "hot_tier", None)
    atoms = {}
    if hot is not None:
        atoms.update(getattr(hot, "atoms", None) or {})
        atoms.update(getattr(hot, "staging", None) or {})

    count = 0
    for atom in atoms.values():
        if count >= _MAX_ATOMS:
            break
        count += 1

        raw_mtype = getattr(atom, "memory_type", None) or (atom.get("memory_type") if isinstance(atom, dict) else None)
        mtype_val = getattr(raw_mtype, "value", raw_mtype)
        mtype = str(mtype_val or "").lower()
        meta = getattr(atom, "metadata", None) or (atom.get("metadata") if isinstance(atom, dict) else None) or {}
        if mtype == "skill" or meta.get("skill_name"):
            types.add("skill")
            sk = meta.get("skill_name") or getattr(atom, "id", None) or (atom.get("id") if isinstance(atom, dict) else None)
            if sk:
                entity_types.setdefault(str(sk), set()).add("skill")
        elif mtype == "profile" or meta.get("user_id"):
            types.add("person")
            uid = meta.get("user_id")
            if uid:
                entity_types.setdefault(str(uid), set()).add("person")

        triples = getattr(atom, "triples", None) or []
        if not triples and isinstance(atom, dict):
            triples = atom.get("triples") or (atom.get("metadata") or {}).get("triples") or []
        for triple in triples:
            if not triple or len(triple) < 3:
                continue
            s, p, o = str(triple[0]), _pred_key(triple[1]), str(triple[2])
            if p:
                predicates.add(p)
            if p == "is_skill":
                types.add("skill")
                entity_types.setdefault(s, set()).add("skill")
                entity_types.setdefault(o, set()).add("skill")
                continue
            if p not in TYPE_DEFINING_PREDICATES:
                continue
            # Only compact type ids — skip sentence-length objects
            tid = normalize_type(o, default="")
            if not tid or tid == "entity":
                # Still accept short open ids not in seed
                slug = (
                    o.strip()
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                if (
                    slug
                    and len(slug) <= 64
                    and slug.replace("_", "").isalnum()
                    and not any(ch.isdigit() for ch in slug[:1])
                ):
                    tid = slug
                else:
                    continue
            types.add(tid)
            entity_types.setdefault(s, set()).add(tid)

    return {
        "types": sorted(types),
        "predicates": sorted(predicates),
        "entity_types": {k: sorted(v) for k, v in entity_types.items()},
        "source": "hot_tier_scan",
    }


def discover_types(engine: Any) -> Dict[str, Any]:
    """Return {types, predicates, entity_types} from the live engine."""
    if engine is None:
        return {"types": [], "predicates": [], "entity_types": {}, "source": "none"}

    getter = getattr(engine, "get_discovered_types", None)
    if callable(getter):
        try:
            raw = getter()
            if isinstance(raw, dict):
                return {
                    "types": list(raw.get("types") or []),
                    "predicates": list(raw.get("predicates") or []),
                    "entity_types": dict(raw.get("entity_types") or {}),
                    "source": "engine",
                }
        except Exception as e:
            logger.warning("engine.get_discovered_types failed: %s", e)

    try:
        return _scan_hot_tier(engine)
    except Exception as e:
        logger.warning("hot-tier type discovery failed: %s", e)
        return {"types": [], "predicates": [], "entity_types": {}, "source": "error"}
