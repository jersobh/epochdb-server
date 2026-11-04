"""
Suggested entity-type seed catalog for apps (e.g. Percepta).

EpochDB auto-discovers types from triples. Apps may predefine or create types
via the type registry (see type_registry.py). This module is a convenient
default seed — not a hard clamp on discovery.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

# Suggested seed (~22). Apps POST these via /entity_types/seed.
ENTITY_TYPES: List[str] = [
    "person",
    "organization",
    "role",
    "agent",
    "product",
    "service",
    "feature",
    "artifact",
    "system",
    "dataset",
    "place",
    "event",
    "market",
    "channel",
    "campaign",
    "concept",
    "skill",
    "process",
    "metric",
    "evidence",
    "hypothesis",
    "session",
    "project",
    "goal",
    "risk",
    "issue",
    "decision",
    "outcome",
    "objective",
    "constraint",
    "assumption",
    "dependency",
]

ENTITY_TYPE_SET: Set[str] = set(ENTITY_TYPES)

ENTITY_TYPE_LABELS: Dict[str, str] = {
    "person": "Person",
    "organization": "Organization",
    "role": "Role",
    "agent": "Agent",
    "product": "Product",
    "service": "Service",
    "feature": "Feature",
    "artifact": "Artifact",
    "system": "System",
    "dataset": "Dataset",
    "place": "Place",
    "event": "Event",
    "market": "Market",
    "channel": "Channel",
    "campaign": "Campaign",
    "concept": "Concept",
    "skill": "Skill",
    "process": "Process",
    "metric": "Metric",
    "evidence": "Evidence",
    "hypothesis": "Hypothesis",
    "session": "Session",
    "entity": "Entity",
    "project": "Project",
    "goal": "Goal",
    "risk": "Risk",
    "issue": "Issue",
    "decision": "Decision",
    "outcome": "Outcome",
    "objective": "Objective",
    "constraint": "Constraint",
    "assumption": "Assumption",
    "dependency": "Dependency",
}

# Soft aliases used when mapping free-text into a *registered* type id.
TYPE_ALIASES: Dict[str, str] = {
    "people": "person",
    "human": "person",
    "contact": "person",
    "user": "person",
    "company": "organization",
    "org": "organization",
    "employer": "organization",
    "team": "Team",
    "community": "organization",
    "job_title": "role",
    "title": "role",
    "position": "role",
    "bot": "agent",
    "ai_agent": "agent",
    "sku": "product",
    "offering": "product",
    "tool": "system",
    "platform": "system",
    "software": "system",
    "infra": "system",
    "infrastructure": "system",
    "model": "system",
    "api": "system",
    "doc": "artifact",
    "document": "artifact",
    "post": "artifact",
    "article": "artifact",
    "paper": "artifact",
    "repo": "artifact",
    "repository": "artifact",
    "media": "artifact",
    "location": "place",
    "city": "place",
    "country": "place",
    "venue": "place",
    "meetup": "event",
    "conference": "event",
    "interview": "event",
    "vertical": "market",
    "segment": "market",
    "industry": "market",
    "social_network": "channel",
    "email": "channel",
    "call": "channel",
    "topic": "concept",
    "domain": "concept",
    "idea": "concept",
    "competency": "skill",
    "workflow": "process",
    "playbook": "process",
    "kpi": "metric",
    "fact": "evidence",
    "observation": "evidence",
    "inference": "hypothesis",
    "claim": "hypothesis",
    "observe_session": "session",
    "experience": "role",
    "headline": "artifact",
    "about": "artifact",
    "visible_post": "artifact",
    "visible_quote": "artifact",
    "post_themes": "concept",
    "post_theme": "concept",
    "followers": "metric",
    "follower_count": "metric",
    "education": "artifact",
    "current_employer": "organization",
    "lead": "role",
    "rule": "process",
    "step": "process",
    "stakeholder": "person",
    "champion": "person",
    "economic_buyer": "person",
    "technical_gatekeeper": "person",
}

TYPE_DEFINING_PREDICATES: Set[str] = {
    "is_a",
    "instance_of",
    "rdf:type",
    "type",
    "has_type",
    "entity_type",
    "classified_as",
    "subclass_of",
    "kind_of",
}

PRED_ROLES: Dict[str, tuple] = {
    "born_in": ("person", "place"),
    "lives_in": ("person", "place"),
    "died_in": ("person", "place"),
    "works_at": ("person", "organization"),
    "worked_at": ("person", "organization"),
    "employed_by": ("person", "organization"),
    "studied_at": ("person", "organization"),
    "member_of": ("person", "organization"),
    "leads": ("person", "organization"),
    "founded": ("person", "organization"),
    "founded_by": ("organization", "person"),
    "collaborated_with": ("person", "person"),
    "won": ("person", "event"),
    "wrote": ("person", "artifact"),
    "authored": ("person", "artifact"),
    "mentioned_in": ("person", "artifact"),
    "participated_in": ("person", "event"),
    "published_on": ("artifact", "channel"),
    "about": (None, "concept"),
    "holds_role": ("person", "role"),
    "has_label": ("person", None),
    "observed": ("person", "evidence"),
    "hypothesized": ("person", "hypothesis"),
    "works_on": ("person", "product"),
    "builds": ("person", "product"),
    "sells": ("organization", "product"),
    "offers": ("organization", "service"),
    "has_feature": ("product", "feature"),
    "uses": ("person", "system"),
    "runs_on": ("system", "system"),
    "located_in": (None, "place"),
    "headquartered_in": ("organization", "place"),
    "competes_with": ("organization", "organization"),
    "part_of": (None, "organization"),
    "invented": ("person", "product"),
    "commercialized": ("organization", "system"),
    "powers": ("product", "system"),
    "developed": ("person", "concept"),
    "created": ("person", "artifact"),
    "designed": ("person", "artifact"),
    "discovered": ("person", "concept"),
    "formulated": ("person", "concept"),
    "patented": ("person", "product"),
    "produces": ("organization", "product"),
    "manufactures": ("organization", "product"),
    "enables": ("concept", "system"),
    "influences": ("concept", "system"),
    "implements": ("system", "concept"),
    "distributes_data_to": ("system", "system"),
    "routes_queries_to": ("system", "system"),
    "compresses_epochs_into": ("system", "artifact"),
    "supports_queries_via": ("artifact", "system"),
    "analyzes": ("system", "system"),
    "generates": ("system", "dataset"),
    "combines": ("system", "dataset"),
    "is_skill": ("skill", "skill"),
    "has_step": ("skill", "process"),
    "wrote_first_algorithm_for": ("person", "artifact"),
    "named_after": (None, "person"),
    "led_by": ("organization", "person"),
    "ceo_of": ("person", "organization"),
    "co_founded": ("person", "organization"),
    "pioneered": ("person", "concept"),
    "fellow_at": ("person", "organization"),
    "rival_of": ("person", "person"),
    "mentored": ("person", "person"),
    "cracked": ("person", "system"),
    "built": ("organization", "system"),
    "manages": ("system", "system"),
    "powered_by": ("product", "product"),
    "launches": ("product", "system"),
    "provides": ("system", "service"),
    "connects_to": ("service", "system"),
    "supports": ("organization", "system"),
}


def _slug(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def normalize_type(
    raw: Optional[str],
    *,
    allowed: Optional[Set[str]] = None,
    default: str = "entity",
) -> str:
    """
    Normalize a type label.

    If ``allowed`` is set (registered ∪ discovered), map aliases into that set.
    Otherwise allow any snake_case id (open discovery) with soft alias rewrite
    into the suggested seed when applicable.
    """
    if raw is None:
        return default
    s = _slug(raw)
    if not s:
        return default
    # Reject long free-text (job titles, sentences) as type ids
    if len(s) > 64 or ":" in str(raw):
        if ":" in str(raw):
            prefix = _slug(str(raw).split(":", 1)[0])
            return normalize_type(prefix, allowed=allowed, default=default)
        return default

    if allowed is not None:
        if s in allowed:
            return s
        alias = TYPE_ALIASES.get(s)
        if alias and alias in allowed:
            return alias
        return default

    if s in ENTITY_TYPE_SET:
        return s
    if s in TYPE_ALIASES:
        return TYPE_ALIASES[s]
    # Open: accept compact identifiers
    if s.replace("_", "").isalnum() and len(s) <= 64:
        return s
    return default


def infer_type_from_name(
    name: str, *, allowed: Optional[Set[str]] = None
) -> Optional[str]:
    n = str(name or "").strip()
    if not n:
        return None
    low = n.lower()
    candidates = []
    if low.startswith("person_") or low == "person":
        candidates.append("person")
    if low.startswith("name_") or low.startswith("email_") or low.startswith("phone_"):
        candidates.append("person")
    if low.startswith("org_") or low.startswith("company_"):
        candidates.append("organization")
    if any(k in low for k in (" university", " college", " institute", " foundation", " electric", " corp", " inc", " ltd", " gmbh")):
        candidates.append("organization")
    if low.startswith("product_"):
        candidates.append("product")
    if low.startswith("session_"):
        candidates.append("session")
    if low.startswith("skill:") or low.startswith("skill "):
        candidates.append("skill")
    if low in ("evidence", "hypothesis"):
        candidates.append(low)
    if low.startswith("social_network:") or low.startswith("social_network "):
        candidates.append("channel")
    if ":" in n:
        candidates.append(_slug(n.split(":", 1)[0]))

    for c in candidates:
        mapped = normalize_type(c, allowed=allowed, default="")
        if mapped and mapped != "entity":
            return mapped
    return None


def seed_entries() -> List[Dict[str, str]]:
    return [
        {
            "id": t,
            "label": ENTITY_TYPE_LABELS.get(t, t.replace("_", " ").title()),
            "description": f"Seed type: {t}",
        }
        for t in ENTITY_TYPES
    ]


def ontology_payload() -> Dict[str, object]:
    """Backward-compatible suggested catalog (not the live registry). """
    return {
        "types": list(ENTITY_TYPES),
        "labels": dict(ENTITY_TYPE_LABELS),
        "seed": True,
        "closed": False,
    }
