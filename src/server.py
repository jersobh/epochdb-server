# src/server.py
import logging
import os
import sys
import uvicorn
import uuid
import hashlib
import bisect
import asyncio
import httpx
import json
import time
import numpy as np
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union, Tuple, Set

# Compatibility alias for sentence-transformers 5.x with 6.x models
try:
    import sentence_transformers.sentence_transformer.modules.normalize as _st_norm
    sys.modules.setdefault("sentence_transformers.base.modules.normalize", _st_norm)
except Exception:
    pass

class EpochEmbedderAdapter:
    """
    Adapter for SentenceTransformer and other embedding models.
    Enforces:
    1. Instruction Prefix for inference/query vs document passages (e.g. Qwen3-Embedding).
    2. EOS / Last-token pooling verification for causal decoder-only models.
    3. Matryoshka dimension truncation and L2 re-normalization.
    """
    def __init__(self, model: Any, dim: int, model_name: str = ""):
        self.model = model
        self.dim = dim
        self.model_name = model_name
        self._verify_pooling()

    def _verify_pooling(self):
        """Verify and enforce last-token pooling for causal decoder-only architectures."""
        try:
            modules = getattr(self.model, "_modules", {})
            for mod in modules.values():
                if mod.__class__.__name__ == "Pooling" or hasattr(mod, "pooling_mode"):
                    current_mode = getattr(mod, "pooling_mode", None)
                    is_causal = False
                    for m in modules.values():
                        auto_mod = getattr(m, "auto_model", None)
                        arch = getattr(auto_mod, "config", None)
                        if arch:
                            arch_str = str(getattr(arch, "architectures", "")).lower()
                            model_type = str(getattr(arch, "model_type", "")).lower()
                            if getattr(arch, "is_decoder", False) or "causal" in arch_str or "qwen" in model_type or "llama" in model_type:
                                is_causal = True
                                break
                    if not is_causal and any(k in self.model_name.lower() for k in ["qwen"]):
                        is_causal = True

                    if is_causal and current_mode != "lasttoken":
                        logging.getLogger("epochdb_server").warning(
                            f"[EpochEmbedderAdapter] Causal decoder model detected with pooling_mode='{current_mode}'. "
                            f"Enforcing 'lasttoken' pooling for correct semantic similarity."
                        )
                        if hasattr(mod, "pooling_mode"):
                            mod.pooling_mode = "lasttoken"
                        if hasattr(mod, "pooling_mode_lasttoken"):
                            mod.pooling_mode_lasttoken = True
                        if hasattr(mod, "pooling_mode_mean_tokens"):
                            mod.pooling_mode_mean_tokens = False
        except Exception as e:
            logging.getLogger("epochdb_server").debug(f"[EpochEmbedderAdapter] Could not check pooling: {e}")

    def _get_query_prompt_kwargs(self) -> dict:
        custom_inst = os.getenv("EPOCHDB_QUERY_INSTRUCTION")
        if custom_inst:
            if not custom_inst.endswith("Query: ") and not custom_inst.endswith("Query:"):
                custom_inst = f"{custom_inst.rstrip()}\nQuery: "
            return {"prompt": custom_inst}
        prompts = getattr(self.model, "prompts", None)
        if isinstance(prompts, dict) and "query" in prompts:
            return {"prompt_name": "query"}
        model_id = str(getattr(self.model, "model_name_or_path", "") or self.model_name).lower()
        if "epochdb-embed" in model_id or "e5-" in model_id:
            return {"prompt": "query: "}
        if any(k in model_id for k in ["qwen", "bge-"]):
            return {"prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "}
        return {}

    def encode(
        self,
        text: Union[str, List[str]],
        is_query: bool = False,
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        target_dim = truncate_dim or self.dim
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension", None)
        native_dim = dim_fn() if callable(dim_fn) else None

        encode_kwargs = dict(kwargs)
        encode_kwargs["normalize_embeddings"] = normalize_embeddings

        if prompt is not None:
            encode_kwargs["prompt"] = prompt
        elif prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name
        elif is_query:
            encode_kwargs.update(self._get_query_prompt_kwargs())
        else:
            prompts = getattr(self.model, "prompts", None)
            if isinstance(prompts, dict) and prompts.get("document"):
                encode_kwargs["prompt_name"] = "document"

        # Apply Matryoshka truncation if target_dim < native_dim
        if target_dim and native_dim and target_dim < native_dim:
            encode_kwargs["truncate_dim"] = target_dim

        try:
            res = self.model.encode(text, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("truncate_dim", None)
            encode_kwargs.pop("prompt_name", None)
            encode_kwargs.pop("prompt", None)
            res = self.model.encode(text, **encode_kwargs)

        def _post_process(vec):
            v = np.array(vec, dtype=np.float32)
            if target_dim and len(v) > target_dim:
                v = v[:target_dim]
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v = v / norm
            return v

        if isinstance(text, str):
            return _post_process(res)
        return np.array([_post_process(v) for v in res], dtype=np.float32)

    def encode_batch(self, texts: List[str], is_query: bool = False, **kwargs) -> np.ndarray:
        return self.encode(texts, is_query=is_query, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)

# Apply runtime monkey patches to EpochDB to guarantee embedding alignment and avoid OOM from multiple copies
try:
    import threading
    from epochdb.api.db import EpochDB, Memory
    from epochdb.engine import EpochDB as EngineEpochDB

    _SHARED_EMBEDDER_CACHE: Dict[str, Any] = {}
    _SHARED_EMBEDDER_LOCK = threading.Lock()

    _orig_get_embedder = EngineEpochDB._get_embedder
    def _patched_get_embedder(self):
        if getattr(self, "_embedder", None) is not None:
            return self._embedder

        model_key = (self._model_name or "").strip()
        with _SHARED_EMBEDDER_LOCK:
            if model_key in _SHARED_EMBEDDER_CACHE:
                cached = _SHARED_EMBEDDER_CACHE[model_key]
                self._embedder = cached
                return cached

            embedder = _orig_get_embedder(self)
            if embedder is not None and not isinstance(embedder, EpochEmbedderAdapter):
                wrapped = EpochEmbedderAdapter(embedder, dim=self.dim, model_name=self._model_name or "")
                _SHARED_EMBEDDER_CACHE[model_key] = wrapped
                self._embedder = wrapped
                return wrapped
            elif embedder is not None:
                _SHARED_EMBEDDER_CACHE[model_key] = embedder
                self._embedder = embedder
                return embedder
            return embedder
    EpochDB._get_embedder = _patched_get_embedder
    EngineEpochDB._get_embedder = _patched_get_embedder

    _orig_query = EpochDB.query
    def _patched_query(self, text: str, k: int = 5, filters: Optional[dict] = None, min_score: float = 0.0, memory_type: Optional[str] = None, context_window: int = 0, expand_hops: int = 0, cold_search_mode: Optional[str] = None) -> List[Memory]:
        query_entities = [str(e) for e in self.extract_entities(text)]
        if self._model_name:
            embedder = self._get_embedder()
            try:
                raw_emb = embedder.encode(text, is_query=True, normalize_embeddings=True)
            except TypeError:
                raw_emb = embedder.encode(text, normalize_embeddings=True)
            emb = np.array(raw_emb, dtype=np.float32)
            if len(emb) > self.dim:
                emb = emb[:self.dim]
                norm = np.linalg.norm(emb)
                if norm > 1e-10:
                    emb /= norm
        else:
            emb = np.zeros(self.dim, dtype=np.float32)

        with self._internal_lock:
            atoms = self.retriever.search(
                query_emb=emb,
                top_k=k,
                expand_hops=max(0, int(expand_hops or 0)),
                query_entities=query_entities,
                filters=filters,
                context_window=context_window,
                cold_search_mode=cold_search_mode,
                query_text=text,
            )
            if memory_type:
                from epochdb.core.atom import MemoryType
                try:
                    mt = MemoryType(memory_type)
                    atoms = [a for a in atoms if a.memory_type == mt]
                except ValueError:
                    pass
            memories = []
            for atom in atoms:
                mem = Memory(atom)
                if hasattr(atom, "score"):
                    mem.score = atom.score
                memories.append(mem)
            return memories
    EpochDB.query = _patched_query
    EngineEpochDB.query = _patched_query

    # Patch supersession in RetrievalManager and ColdTier so co-occurrence
    # and biographical relationship triples do not falsely supersede earlier facts
    try:
        import textwrap, inspect
        import epochdb.retrieval.retrieval as ret_mod
        from epochdb.retrieval.retrieval import RetrievalManager

        src_rm = textwrap.dedent(inspect.getsource(RetrievalManager.search))
        old_rm_block = """    for atom, _ in recency_sorted:
        for s, p, o in atom.triples:
            state_key = (str(s).lower(), str(p).lower())"""
        new_rm_block = """    NON_SUPERSEDING_PREDICATES = {
        'co_occurs_with', 'mentions', 'associated_with', 'related_to',
        'tagged_with', 'has_tag', 'contains', 'part_of', 'member_of',
        'founded', 'co_founded', 'created', 'invented', 'designed',
        'mentored', 'worked_at', 'attended', 'developed', 'pioneered',
        'authored', 'discovered', 'produced', 'competes_with',
    }
    for atom, _ in recency_sorted:
        for s, p, o in atom.triples:
            p_l = str(p).lower()
            if p_l in NON_SUPERSEDING_PREDICATES:
                continue
            state_key = (str(s).lower(), p_l)"""
        if old_rm_block in src_rm:
            patched_rm_src = src_rm.replace(old_rm_block, new_rm_block, 1)
            scope_rm = ret_mod.__dict__.copy()
            exec(patched_rm_src, scope_rm)
            RetrievalManager.search = scope_rm['search']
            logging.getLogger("epochdb_server").info("Successfully patched RetrievalManager.search supersession logic.")
    except Exception as _rm_err:
        logging.getLogger("epochdb_server").warning(f"Could not patch RetrievalManager: {_rm_err}")

    try:
        import textwrap, inspect
        import epochdb.storage.cold_tier as cold_mod
        from epochdb.storage.cold_tier import ColdTier

        src_c = textwrap.dedent(inspect.getsource(ColdTier.compact))
        old_c_block = """        is_superseded = False
        for s, p, o in atom.triples:
            state_key = (str(s).lower(), str(p).lower())"""
        new_c_block = """        NON_SUPERSEDING_PREDICATES = {
            'co_occurs_with', 'mentions', 'associated_with', 'related_to',
            'tagged_with', 'has_tag', 'contains', 'part_of', 'member_of',
            'founded', 'co_founded', 'created', 'invented', 'designed',
            'mentored', 'worked_at', 'attended', 'developed', 'pioneered',
            'authored', 'discovered', 'produced', 'competes_with',
        }
        is_superseded = False
        for s, p, o in atom.triples:
            p_l = str(p).lower()
            if p_l in NON_SUPERSEDING_PREDICATES:
                continue
            state_key = (str(s).lower(), p_l)"""
        if old_c_block in src_c:
            patched_c_src = src_c.replace(old_c_block, new_c_block, 1)
            scope_c = cold_mod.__dict__.copy()
            exec(patched_c_src, scope_c)
            ColdTier.compact = scope_c['compact']
            logging.getLogger("epochdb_server").info("Successfully patched ColdTier.compact supersession logic.")
    except Exception as _cold_err:
        logging.getLogger("epochdb_server").warning(f"Could not patch ColdTier: {_cold_err}")
except Exception as _patch_err:
    logging.getLogger("epochdb_server").warning(f"Could not monkey-patch EpochDB: {_patch_err}")

from fastapi import FastAPI, HTTPException, status, Security, Depends, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union, Tuple, Set

from epochdb import AsyncEpochDB
from fastapi import Header, Query
from sse_starlette.sse import EventSourceResponse

try:
    from auth import verify_scoped_auth, Permission, get_keystore, ScopedAPIKey
except ImportError:
    from src.auth import verify_scoped_auth, Permission, get_keystore, ScopedAPIKey

try:
    from ontology import (
        ENTITY_TYPES,
        ENTITY_TYPE_SET,
        PRED_ROLES,
        TYPE_DEFINING_PREDICATES,
        _slug,
        infer_type_from_name,
        normalize_type,
        seed_entries,
    )
    from type_registry import (
        delete_type as registry_delete_type,
        get_registered_ids,
        list_registered,
        seed_types as registry_seed_types,
        upsert_type as registry_upsert_type,
    )
    from type_discovery import discover_types
except ImportError:
    from src.ontology import (
        ENTITY_TYPES,
        ENTITY_TYPE_SET,
        PRED_ROLES,
        TYPE_DEFINING_PREDICATES,
        _slug,
        infer_type_from_name,
        normalize_type,
        seed_entries,
    )
    from src.type_registry import (
        delete_type as registry_delete_type,
        get_registered_ids,
        list_registered,
        seed_types as registry_seed_types,
        upsert_type as registry_upsert_type,
    )
    from src.type_discovery import discover_types

# Export variables for backward compatibility and tests
API_KEY = os.getenv("API_KEY")
INTERNAL_AUTH_TOKEN = os.getenv("INTERNAL_AUTH_TOKEN")
SERVER_VERSION = "0.14.1"
# Coordinator→shard HTTP timeout. Embeddings on small VPS CPUs often exceed 30s
# under concurrent ingest; default matches gunicorn --timeout 120.
COORDINATOR_HTTP_TIMEOUT = float(os.getenv("COORDINATOR_HTTP_TIMEOUT", "120"))
# Short timeout for graph fan-out so one slow shard returns a partial graph
# instead of holding the coordinator for the full write timeout.
COORDINATOR_GRAPH_TIMEOUT = float(os.getenv("COORDINATOR_GRAPH_TIMEOUT", "25"))
# Bound CPU-bound sentence-transformer work per shard process. A full queue is
# rejected rather than letting requests pile up until coordinator timeouts.
SHARD_EMBED_CONCURRENCY = max(1, int(os.getenv("SHARD_EMBED_CONCURRENCY", "1")))
SHARD_EMBED_QUEUE_LIMIT = max(0, int(os.getenv("SHARD_EMBED_QUEUE_LIMIT", "8")))
# Pre-warm this tenant/namespace at shard boot (Percepta uses global/default).
# Empty → skip; avoids a second MiniLM cold-load on the first app request.
DEFAULT_TENANT = (os.getenv("DEFAULT_TENANT") or "").strip() or None
DEFAULT_NAMESPACE = (os.getenv("DEFAULT_NAMESPACE") or "").strip() or None
_embed_semaphore: Optional[asyncio.Semaphore] = None
_embed_waiters = 0
# Last successful full /stats snapshot — served by ?probe=1 without touching the DB.
_LAST_SHARD_DB_STATS: Dict[str, Any] = {
    "memory_count": 0,
    "l1_size": 0,
    "l2_size": 0,
    "entity_count": 0,
}


def format_shard_error(resp: Any) -> str:
    """Readable shard failure text (TimeoutError/ReadError often have empty str())."""
    if isinstance(resp, Exception):
        msg = str(resp).strip()
        return msg if msg else repr(resp)
    if isinstance(resp, httpx.Response):
        body = (resp.text or "").strip().replace("\n", " ")[:200]
        return f"HTTP {resp.status_code}" + (f": {body}" if body else "")
    return repr(resp)


def should_mark_shard_offline(resp: Any) -> bool:
    """Only mark offline for hard connectivity loss — not overload timeouts."""
    return isinstance(resp, httpx.ConnectError)


# -------------------------------------------------------------------------
# 1. Structured Logging Configuration
# -------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s (Line: %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("epochdb_production_server")

# httpx logs one INFO line per outbound request; with 5s health/stats polling
# across shards this floods coordinator logs and buries real errors.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# -------------------------------------------------------------------------
# Dynamic CausalLM support for HuggingFace triple extraction
# (e.g. Qwen3.5, Llama, Mistral such as jersobh/epochdb-extractor-0.8b)
# -------------------------------------------------------------------------
def _patch_epochdb_fact_extractor():
    try:
        import re
        from epochdb.core import fact_extractor as fe
        if getattr(fe, "_causal_lm_patched", False):
            return

        orig_load = fe.HuggingFaceFactExtractor._load
        orig_extract = fe.HuggingFaceFactExtractor.extract

        _SHARED_EXTRACTOR_MODELS: Dict[str, Any] = {}
        _SHARED_EXTRACTOR_LOCK = threading.Lock()

        def patched_load(self):
            if self._model is not None:
                return

            cache_dir = self._cache_dir()
            load_id = cache_dir if cache_dir and os.path.isdir(cache_dir) and os.listdir(cache_dir) else self.model_id

            with _SHARED_EXTRACTOR_LOCK:
                if load_id in _SHARED_EXTRACTOR_MODELS:
                    cached_model, cached_tok, is_gl, is_causal = _SHARED_EXTRACTOR_MODELS[load_id]
                    self._model = cached_model
                    self._tokenizer = cached_tok
                    self._is_gliner = is_gl
                    self._is_causal = is_causal
                    return

                logger.info(f"Loading HuggingFace extraction model: {load_id}")

                # 1. Detect if the checkpoint is a GLiNER zero-shot entity model
                is_gliner = "gliner" in load_id.lower() or "epochdb-extractor" in load_id.lower()
                if not is_gliner:
                    try:
                        from transformers import AutoConfig
                        cfg = AutoConfig.from_pretrained(load_id)
                        if getattr(cfg, "model_type", "") == "gliner_uni_encoder_span":
                            is_gliner = True
                    except Exception as e:
                        if "gliner" in str(e).lower():
                            is_gliner = True

                if is_gliner:
                    try:
                        from gliner import GLiNER
                        self._model = GLiNER.from_pretrained(load_id)
                        self._is_gliner = True
                        self._is_causal = False
                        self._tokenizer = None
                        if cache_dir and load_id == self.model_id:
                            try:
                                os.makedirs(cache_dir, exist_ok=True)
                                self._model.save_pretrained(cache_dir)
                            except Exception as ce:
                                logger.warning(f"Could not cache GLiNER model to {cache_dir}: {ce}")
                        _SHARED_EXTRACTOR_MODELS[load_id] = (self._model, None, True, False)
                        logger.info(f"Loaded GLiNER extraction model successfully: {load_id}")
                        return
                    except Exception as gliner_err:
                        logger.warning(f"Could not load as GLiNER model: {gliner_err}")

                # 2. Otherwise load via Transformers (Seq2Seq, CausalLM, or GLiNER fallback)
                try:
                    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                except ImportError:
                    raise ImportError(
                        "transformers is required for Hugging Face triple extraction. "
                        "Install it with: pip install epochdb[extraction]"
                    )

                self._tokenizer = AutoTokenizer.from_pretrained(load_id)
                try:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(load_id)
                    self._is_causal = False
                except (ValueError, KeyError, Exception):
                    try:
                        import torch
                        from transformers import AutoModelForCausalLM
                        self._model = AutoModelForCausalLM.from_pretrained(
                            load_id,
                            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        )
                        self._is_causal = True
                    except Exception as causal_err:
                        if "gliner" in str(causal_err).lower():
                            from gliner import GLiNER
                            self._model = GLiNER.from_pretrained(load_id)
                            self._is_gliner = True
                            self._is_causal = False
                            _SHARED_EXTRACTOR_MODELS[load_id] = (self._model, None, True, False)
                            logger.info(f"Loaded GLiNER extraction model after transformers mismatch: {load_id}")
                            return
                        raise causal_err
                self._model.eval()

                if cache_dir and load_id == self.model_id:
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                        self._tokenizer.save_pretrained(cache_dir)
                        self._model.save_pretrained(cache_dir)
                    except Exception as e:
                        logger.warning(f"Could not cache HF extraction model to {cache_dir}: {e}")

                _SHARED_EXTRACTOR_MODELS[load_id] = (
                    self._model,
                    self._tokenizer,
                    getattr(self, "_is_gliner", False),
                    getattr(self, "_is_causal", False),
                )

        def patched_extract(self, text: str):
            self._load()

            if getattr(self, "_is_gliner", False):
                raw_labels = os.getenv(
                    "EPOCHDB_EXTRACTION_LABELS",
                    "person, organization, company, location, product, concept, event, role, technology, topic"
                )
                labels = [l.strip() for l in raw_labels.split(",") if l.strip()]
                threshold = float(os.getenv("EPOCHDB_EXTRACTION_THRESHOLD", "0.4"))
                try:
                    ents = self._model.predict_entities(text, labels=labels, threshold=threshold)
                except Exception as ex:
                    logger.warning(f"GLiNER prediction error: {ex}")
                    return []

                triples = []
                cleaned_entities = []
                seen_entities = set()
                for e in ents:
                    ent_text = (e.get("text") or "").strip()
                    ent_label = (e.get("label") or "").strip()
                    if not ent_text:
                        continue
                    if ent_label:
                        triples.append((ent_text, "is_a", ent_label))
                    if ent_text.lower() not in seen_entities:
                        seen_entities.add(ent_text.lower())
                        cleaned_entities.append(ent_text)

                # Connect consecutive extracted entities with relational edge
                if len(cleaned_entities) >= 2:
                    for i in range(len(cleaned_entities) - 1):
                        triples.append((cleaned_entities[i], "associated_with", cleaned_entities[i + 1]))
                    if len(cleaned_entities) > 2:
                        triples.append((cleaned_entities[0], "associated_with", cleaned_entities[-1]))
                elif len(cleaned_entities) == 1 and not triples:
                    triples.append((cleaned_entities[0], "mentions", cleaned_entities[0]))

                return fe._normalize_triples(triples)

            import torch
            assert self._tokenizer is not None and self._model is not None

            if getattr(self, "_is_causal", False):
                prompt = (
                    "Extract relationship triples from the text as JSON with key "
                    '"triples" as a list of {subject, predicate, object} objects.\n'
                    f"Text: {text}\nJSON:"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(device)
                inputs = self._tokenizer(
                    prompt,
                    max_length=self.max_input_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_length,
                        do_sample=False,
                    )
                decoded = self._tokenizer.batch_decode(
                    generated[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0]
                triples = fe._parse_jsonish_triples(decoded)
                return triples if triples else []
            else:
                return orig_extract(self, text)

        orig_parse = fe._parse_jsonish_triples
        def patched_parse_jsonish_triples(text: str):
            res = orig_parse(text)
            if res:
                return res
            import ast, re
            for match in re.finditer(r"(\{.*\}|\[.*\])", text, re.DOTALL):
                try:
                    data = ast.literal_eval(match.group(1).strip())
                    if isinstance(data, dict):
                        got = fe._normalize_triples(data.get("triples") or data.get("relations") or [])
                    else:
                        got = fe._normalize_triples(data)
                    if got:
                        return got
                except Exception:
                    continue
            return []

        fe.HuggingFaceFactExtractor._load = patched_load
        fe.HuggingFaceFactExtractor.extract = patched_extract
        fe._parse_jsonish_triples = patched_parse_jsonish_triples

        orig_fe_extract = fe.FactExtractor.extract
        def patched_fe_extract(self, text: str):
            if getattr(self, "_failed_once", False):
                return self._local_extractor.extract(text)
            try:
                return self._get_impl().extract(text)
            except Exception as e:
                logger.warning(f"Falling back to LocalFactExtractor permanently for this session ({self._backend_name}): {e}")
                self._failed_once = True
                self._impl = self._local_extractor
                return self._local_extractor.extract(text)

        fe.FactExtractor.extract = patched_fe_extract
        fe._causal_lm_patched = True
        logger.info("Initialized dynamic CausalLM loader for HuggingFace fact extractors.")
    except Exception as ex:
        logger.debug(f"Could not apply CausalLM patch to fact extractor: {ex}")

_patch_epochdb_fact_extractor()


@asynccontextmanager
async def embedding_slot():
    """Reserve bounded shard-local capacity for an embedding operation."""
    global _embed_semaphore, _embed_waiters
    if _embed_semaphore is None:
        _embed_semaphore = asyncio.Semaphore(SHARD_EMBED_CONCURRENCY)

    if _embed_semaphore.locked() and _embed_waiters >= SHARD_EMBED_QUEUE_LIMIT:
        raise HTTPException(
            status_code=503,
            detail="Embedding capacity is saturated; retry shortly.",
            headers={"Retry-After": "5"},
        )

    _embed_waiters += 1
    try:
        async with _embed_semaphore:
            yield
    finally:
        _embed_waiters -= 1


async def run_embedding(fn, *args, **kwargs):
    """Run a CPU-bound embedding with bounded shard-local queueing."""
    async with embedding_slot():
        return await asyncio.to_thread(fn, *args, **kwargs)


# -------------------------------------------------------------------------
# System Resource Profiling Helpers
# -------------------------------------------------------------------------
_last_cpu_times = {"total": 0.0, "idle": 0.0}

def get_cpu_usage() -> float:
    global _last_cpu_times
    try:
        with open("/proc/stat", "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("cpu "):
                parts = [float(x) for x in line.split()[1:]]
                idle = parts[3] + parts[4]
                total = sum(parts)
                
                diff_total = total - _last_cpu_times["total"]
                diff_idle = idle - _last_cpu_times["idle"]
                
                _last_cpu_times["total"] = total
                _last_cpu_times["idle"] = idle
                
                if diff_total <= 0:
                    return 0.0
                return round(max(0.0, min(100.0, (1.0 - diff_idle / diff_total) * 100.0)), 2)
    except Exception:
        pass
    return 0.0

def get_ram_usage() -> Dict[str, float]:
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = float(parts[1])
                    meminfo[key] = val
        total = meminfo.get("MemTotal", 0.0) / 1024.0 / 1024.0
        free = meminfo.get("MemFree", 0.0) / 1024.0 / 1024.0
        buffers = meminfo.get("Buffers", 0.0) / 1024.0 / 1024.0
        cached = meminfo.get("Cached", 0.0) / 1024.0 / 1024.0
        available = meminfo.get("MemAvailable", (free + buffers + cached)) / 1024.0 / 1024.0
        used = total - available
        percent = (used / total * 100.0) if total > 0 else 0.0
        return {
            "total": round(total, 2),
            "available": round(available, 2),
            "used": round(used, 2),
            "percent": round(percent, 2)
        }
    except Exception:
        pass
    return {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}

def get_disk_usage(path: str = ".") -> Dict[str, float]:
    try:
        stat = os.statvfs(path)
        total = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        available = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        used = total - available
        percent = (used / total * 100.0) if total > 0 else 0.0
        return {
            "total": round(total, 2),
            "available": round(available, 2),
            "used": round(used, 2),
            "percent": round(percent, 2)
        }
    except Exception:
        pass
    return {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}

# -------------------------------------------------------------------------
# 2. Clustering Configuration & Consistent Hashing Ring
# -------------------------------------------------------------------------
class ConsistentHashRing:
    def __init__(self, nodes: List[str] = None, replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)

    def add_node(self, node: str):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def remove_node(self, node: str):
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)

    def get_node(self, val: str) -> str:
        if not self.ring:
            raise ValueError("No nodes on the ring")
        h = self._hash(val)
        idx = bisect.bisect_right(self.sorted_keys, h)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]

# Read cluster environment configuration
NODE_MODE = os.getenv("NODE_MODE", "shard").lower()
shard_nodes_str = os.getenv("SHARD_NODES", "")

from enum import Enum

class ConsistencyLevel(str, Enum):
    ONE = "one"
    QUORUM = "quorum"
    ALL = "all"

def get_consistency_level(request: Request) -> ConsistencyLevel:
    header_val = request.headers.get("x-consistency-level") or request.query_params.get("consistency")
    if header_val:
        try:
            return ConsistencyLevel(header_val.lower())
        except ValueError:
            pass
    return ConsistencyLevel.ONE

shard_groups = []
if shard_nodes_str:
    for group_str in shard_nodes_str.split(","):
        if group_str.strip():
            replicas = [r.strip() for r in group_str.split("+") if r.strip()]
            if replicas:
                shard_groups.append(replicas)

shard_nodes = [group[0] for group in shard_groups] if shard_groups else []
hash_ring = ConsistentHashRing(shard_nodes) if shard_nodes else None

# Dynamic embedding configuration (defaults to local offline all-MiniLM-L6-v2)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
# Overview graph (no entity_id): hub + 1-hop KG neighbors only — no atom recall / BFS.
GRAPH_HUB_LIMIT = int(os.getenv("GRAPH_HUB_LIMIT", "24"))
GRAPH_OVERVIEW_NODE_LIMIT = int(os.getenv("GRAPH_OVERVIEW_NODE_LIMIT", "200"))


class RequestContext(BaseModel):
    tenant: Optional[str] = None
    namespace: Optional[str] = None
    key_info: ScopedAPIKey

SHARD_METRICS_CACHE: Dict[str, Any] = {}

def get_replica_group_for_node(node: str) -> List[str]:
    for group in shard_groups:
        if node in group:
            return group
    return [node]

def get_healthy_replica_for_group(group: List[str]) -> Optional[str]:
    for node in group:
        metrics = SHARD_METRICS_CACHE.get(node)
        if not metrics or metrics.get("status") == "healthy":
            return node
    return group[0] if group else None

def get_all_healthy_replicas_for_group(group: List[str]) -> List[str]:
    healthy = []
    for node in group:
        metrics = SHARD_METRICS_CACHE.get(node)
        if not metrics or metrics.get("status") == "healthy":
            healthy.append(node)
    if not healthy and group:
        healthy.append(group[0])
    return healthy

# -------------------------------------------------------------------------
# Cache Layer Configuration
# -------------------------------------------------------------------------
# Key: (tenant, namespace) -> state_version (int)
CLUSTER_STATE_VERSIONS: Dict[tuple[str, str], int] = {}

# Key: (tenant, namespace) -> Dict[cache_key, (cached_data, ETag)]
LOCAL_READ_CACHE: Dict[tuple[str, str], Dict[str, tuple[Any, str]]] = {}

def get_state_version(tenant: Optional[str], namespace: Optional[str]) -> int:
    ctx_key = (tenant or "", namespace or "")
    if ctx_key not in CLUSTER_STATE_VERSIONS:
        CLUSTER_STATE_VERSIONS[ctx_key] = 0
    return CLUSTER_STATE_VERSIONS[ctx_key]

def invalidate_cache(tenant: Optional[str], namespace: Optional[str]):
    ctx_key = (tenant or "", namespace or "")
    CLUSTER_STATE_VERSIONS[ctx_key] = CLUSTER_STATE_VERSIONS.get(ctx_key, 0) + 1
    LOCAL_READ_CACHE[ctx_key] = {}
    if namespace not in ("*", "all"):
        for wildcard in ("*", "all"):
            w_key = (tenant or "", wildcard)
            CLUSTER_STATE_VERSIONS[w_key] = CLUSTER_STATE_VERSIONS.get(w_key, 0) + 1
            LOCAL_READ_CACHE[w_key] = {}
    logger.debug(f"Invalidated read cache for tenant='{tenant}' namespace='{namespace}'. New version: {CLUSTER_STATE_VERSIONS[ctx_key]}")

def generate_etag(endpoint: str, payload_dict: dict, state_version: int) -> str:
    # Stable JSON serialization
    serialized = json.dumps(payload_dict, sort_keys=True)
    raw = f"{endpoint}:{serialized}:{state_version}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

async def handle_cached_read(
    endpoint: str,
    payload: Any,
    request: Request,
    response: Response,
    ctx: RequestContext,
    fetch_func
):
    ctx_key = (ctx.tenant or "", ctx.namespace or "")
    state_version = get_state_version(ctx.tenant, ctx.namespace)
    
    # Standardize payload representation
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else (payload if isinstance(payload, dict) else {"_val": payload})
    etag = generate_etag(endpoint, payload_dict, state_version)
    
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
        
    if ctx_key in LOCAL_READ_CACHE and etag in LOCAL_READ_CACHE[ctx_key]:
        response.headers["ETag"] = etag
        return LOCAL_READ_CACHE[ctx_key][etag]
        
    res_data = await fetch_func()
    
    # Don't cache partial fan-out results: they would keep serving degraded
    # answers until the next write invalidation even after shards recover.
    if not (isinstance(res_data, dict) and res_data.get("partial")):
        if ctx_key not in LOCAL_READ_CACHE:
            LOCAL_READ_CACHE[ctx_key] = {}
        LOCAL_READ_CACHE[ctx_key][etag] = res_data
    response.headers["ETag"] = etag
    return res_data

SSE_LISTENERS: List[asyncio.Queue] = []

async def broadcast_sse_event(event_type: str, data: dict):
    """
    Broadcasts an event to all connected SSE clients.
    """
    if not SSE_LISTENERS:
        return
    logger.debug(f"Broadcasting SSE event '{event_type}' to {len(SSE_LISTENERS)} listeners.")
    for queue in list(SSE_LISTENERS):
        await queue.put({"event": event_type, "data": data})

# Track syncing nodes
SYNCING_NODES = set()
BACKGROUND_SYNC_ENABLED = True

async def sync_node_data(recovering_node: str, source_node: str):
    """
    Synchronizes all databases from source_node to recovering_node.
    """
    logger.info(f"Triggering recovery sync for node {recovering_node} from source {source_node}...")
    try:
        headers = {}
        if INTERNAL_AUTH_TOKEN:
            headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
        
        async with httpx.AsyncClient(headers=headers, timeout=120.0) as sync_client:
            res = await sync_client.get(f"{source_node}/admin/databases")
            res.raise_for_status()
            databases = res.json().get("databases", [])
            
            for db_info in databases:
                tenant = db_info.get("tenant")
                namespace = db_info.get("namespace")
                logger.info(f"Syncing database (tenant={tenant}, namespace={namespace}) from {source_node} to {recovering_node}...")
                
                timeline_payload = {
                    "entity_id": None,
                    "start": None,
                    "end": None
                }
                req_headers = {}
                if tenant:
                    req_headers["X-Tenant"] = tenant
                if namespace:
                    req_headers["X-Namespace"] = namespace
                    
                timeline_res = await sync_client.post(
                    f"{source_node}/get_timeline",
                    json=timeline_payload,
                    headers=req_headers
                )
                timeline_res.raise_for_status()
                memories = timeline_res.json().get("memories", [])
                
                reset_res = await sync_client.post(
                    f"{recovering_node}/admin/reset",
                    json={"tenant": tenant, "namespace": namespace}
                )
                reset_res.raise_for_status()
                
                for m in memories:
                    target_payload = {
                        "text": m.get("payload") or m.get("text"),
                        "metadata": m.get("metadata", {}),
                        "id": m.get("id"),
                        "memory_type": m.get("memory_type")
                    }
                    write_res = await sync_client.post(
                        f"{recovering_node}/remember",
                        json=target_payload,
                        headers=req_headers
                    )
                    write_res.raise_for_status()
                    
            logger.info(f"Sync complete for recovering node {recovering_node} from source {source_node}.")
    except Exception as e:
        logger.error(f"Sync task failed for node {recovering_node} from source {source_node}: {e}")
        raise e

async def poll_shards_loop():
    """
    Background loop that runs on the coordinator to poll shards for health and metrics.
    Employs per-shard exponential backoff and configurable poll intervals to prevent
    hammering shards that are busy with heavy embedding or disk I/O tasks.
    """
    logger.info("Starting background health polling loop for shards...")
    all_replica_nodes = []
    for group in shard_groups:
        all_replica_nodes.extend(group)
    
    stats_timeout = float(os.getenv("COORDINATOR_STATS_TIMEOUT", "15"))
    base_poll_interval = float(os.getenv("COORDINATOR_POLL_INTERVAL", "5"))
    max_backoff = float(os.getenv("COORDINATOR_MAX_BACKOFF", "60"))

    # State tracking per shard: next allowed poll timestamp and failure counts
    next_poll_time: Dict[str, float] = {shard: 0.0 for shard in all_replica_nodes}
    fail_counts: Dict[str, int] = {shard: 0 for shard in all_replica_nodes}
        
    while True:
        if client:
            try:
                now = time.monotonic()
                due_shards = [shard for shard in all_replica_nodes if now >= next_poll_time[shard]]

                if due_shards:
                    # ?probe=1 is CPU/RAM/disk + last-known counts — never opens the
                    # DB or runs embeddings, so polls stay healthy under MiniLM load.
                    tasks = [
                        client.get(f"{shard}/stats", params={"probe": "1"}, timeout=stats_timeout)
                        for shard in due_shards
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    status_changed = False
                    for shard, resp in zip(due_shards, responses):
                        prev_status = SHARD_METRICS_CACHE.get(shard, {}).get("status")
                        if isinstance(resp, httpx.Response) and resp.status_code == 200:
                            fail_counts[shard] = 0
                            next_poll_time[shard] = now + base_poll_interval
                            try:
                                data = resp.json()
                                
                                metrics = {
                                    "status": "healthy",
                                    "memory_count": data.get("memory_count", 0),
                                    "l1_size": data.get("l1_size", 0),
                                    "l2_size": data.get("l2_size", 0),
                                    "entity_count": data.get("entity_count", 0),
                                    "cpu": data.get("cpu", 0.0),
                                    "ram": data.get("ram", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                                    "disk": data.get("disk", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                                }
                                
                                # Trigger background synchronization if node was previously offline/unhealthy
                                logger.debug(f"poll_shards_loop: shard={shard} prev_status={prev_status} syncing={shard in SYNCING_NODES}")
                                if BACKGROUND_SYNC_ENABLED and prev_status not in ("healthy", "synchronizing") and shard not in SYNCING_NODES:
                                    group = get_replica_group_for_node(shard)
                                    source_node = None
                                    for sibling in group:
                                        if sibling != shard:
                                            sib_metrics = SHARD_METRICS_CACHE.get(sibling)
                                            sib_status = sib_metrics.get("status") if sib_metrics else None
                                            logger.debug(f"poll_shards_loop: checking sibling={sibling} status={sib_status}")
                                            if sib_metrics and sib_status == "healthy":
                                                source_node = sibling
                                                break
                                                
                                    logger.info(f"poll_shards_loop: shard={shard} selected source_node={source_node}")
                                    if source_node:
                                        metrics["status"] = "synchronizing"
                                        SHARD_METRICS_CACHE[shard] = metrics
                                        SYNCING_NODES.add(shard)
                                        
                                        async def run_sync_task(node=shard, src=source_node):
                                            try:
                                                await sync_node_data(node, src)
                                                if node in SHARD_METRICS_CACHE:
                                                    SHARD_METRICS_CACHE[node]["status"] = "healthy"
                                                    await broadcast_sse_event("update", {"type": "status_change"})
                                                logger.info(f"Node {node} successfully recovered and synchronized.")
                                            except Exception as e:
                                                if node in SHARD_METRICS_CACHE:
                                                    SHARD_METRICS_CACHE[node]["status"] = "unhealthy"
                                                    await broadcast_sse_event("update", {"type": "status_change"})
                                                logger.error(f"Failed to synchronize recovering node {node}: {e}")
                                            finally:
                                                SYNCING_NODES.discard(node)
                                                
                                        asyncio.create_task(run_sync_task())
                                    else:
                                        logger.info(f"poll_shards_loop: marking shard={shard} healthy immediately because no source_node was found")
                                        SHARD_METRICS_CACHE[shard] = metrics
                                else:
                                    if prev_status == "synchronizing":
                                        metrics["status"] = "synchronizing"
                                    SHARD_METRICS_CACHE[shard] = metrics
                            except Exception as e:
                                SHARD_METRICS_CACHE[shard] = {
                                    "status": "unhealthy",
                                    "error": f"Failed to parse stats: {str(e)}",
                                    "cpu": 0.0,
                                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                }
                        else:
                            # Probe failed/timed out/disconnected
                            fail_counts[shard] += 1
                            err_msg = format_shard_error(resp)

                            if isinstance(resp, httpx.ConnectError):
                                # Hard connectivity loss: cap retry delay to base_poll_interval (e.g. 5s)
                                # so recovered/rebooted nodes are detected promptly.
                                backoff_delay = base_poll_interval
                                next_poll_time[shard] = now + backoff_delay
                                logger.warning(f"poll_shards_loop: shard={shard} unreachable (fails={fail_counts[shard]}, next_poll_in={backoff_delay:.1f}s): {err_msg}")
                                SHARD_METRICS_CACHE[shard] = {
                                    "status": "unhealthy",
                                    "error": err_msg,
                                    "cpu": 0.0,
                                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                }
                            else:
                                # Overload timeout / transient HTTP blips while embedding: exponential backoff
                                # to avoid hammering the busy worker event loop.
                                backoff_delay = min(base_poll_interval * (2 ** (fail_counts[shard] - 1)), max_backoff)
                                next_poll_time[shard] = now + backoff_delay
                                logger.warning(
                                    f"poll_shards_loop: shard={shard} stats probe failed (keeping prior status={prev_status}, fails={fail_counts[shard]}, next_poll_in={backoff_delay:.1f}s): {err_msg}"
                                )
                                if not SHARD_METRICS_CACHE.get(shard):
                                    SHARD_METRICS_CACHE[shard] = {
                                        "status": "healthy",
                                        "error": err_msg,
                                        "cpu": 0.0,
                                        "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                        "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                                    }
                        
                        new_status = SHARD_METRICS_CACHE.get(shard, {}).get("status")
                        if prev_status != new_status:
                            status_changed = True
                    
                    if status_changed:
                        await broadcast_sse_event("update", {"type": "status_change"})
                    
                    if SSE_LISTENERS:
                        total_memory_count = 0
                        total_l1_size = 0
                        total_l2_size = 0
                        total_entity_count = 0
                        
                        group_representatives = []
                        for group in shard_groups:
                            node = get_healthy_replica_for_group(group)
                            if node:
                                group_representatives.append(node)
                                
                        shards_metrics = {}
                        for node in all_replica_nodes:
                            metrics = SHARD_METRICS_CACHE.get(node)
                            if metrics:
                                shards_metrics[node] = metrics
                                if node in group_representatives:
                                    total_memory_count += metrics.get("memory_count", 0)
                                    total_l1_size += metrics.get("l1_size", 0)
                                    total_l2_size += metrics.get("l2_size", 0)
                                    total_entity_count += metrics.get("entity_count", 0)
                                    
                        coord_system = {
                            "cpu": get_cpu_usage(),
                            "ram": get_ram_usage(),
                            "disk": get_disk_usage(os.getenv("STORAGE_DIR", "."))
                        }
                        
                        stats_payload = {
                            "mode": "coordinator",
                            "memory_count": total_memory_count,
                            "l1_size": total_l1_size,
                            "l2_size": total_l2_size,
                            "entity_count": total_entity_count,
                            "system": coord_system,
                            "shards": shards_metrics
                        }
                        
                        await broadcast_sse_event("stats", stats_payload)
            except Exception as e:
                logger.error(f"Error in poll_shards_loop execution: {e}")
        await asyncio.sleep(1)

def get_healthy_shard(key: str) -> str:
    """
    Return the owning shard group for a key, only if it can accept a write.

    Writes must never move to another consistent-hash group: doing so breaks
    deterministic ID routing and leaves data stranded after owner recovery.
    """
    if not shard_nodes:
        raise HTTPException(status_code=500, detail="No shards available to route write request.")
    
    primary_node = hash_ring.get_node(key)
    
    def is_group_good(node: str) -> bool:
        group = get_replica_group_for_node(node)
        for rep in group:
            metrics = SHARD_METRICS_CACHE.get(rep)
            if not metrics:
                return True
            if metrics.get("status") == "healthy":
                if os.getenv("PYTEST_CURRENT_TEST"):
                    return True
                cpu = metrics.get("cpu", 0.0)
                ram_pct = metrics.get("ram", {}).get("percent", 0.0)
                disk_pct = metrics.get("disk", {}).get("percent", 0.0)
                if cpu <= 90.0 and ram_pct <= 90.0 and disk_pct <= 90.0:
                    return True
        return False

    if is_group_good(primary_node):
        return primary_node

    logger.warning(
        "Owner shard group %s cannot accept write for key %r; refusing cross-shard reroute.",
        primary_node, key[:20],
    )
    raise HTTPException(
        status_code=503,
        detail="Owning shard group is unavailable or saturated; retry shortly.",
        headers={"Retry-After": "5"},
    )


# -------------------------------------------------------------------------
# Request Context & Scoped Authentication
# -------------------------------------------------------------------------
def get_context(required_permissions: set[str]):
    async def context_dependency(
        x_tenant: Optional[str] = Header(None, alias="X-Tenant"),
        x_namespace: Optional[str] = Header(None, alias="X-Namespace"),
        tenant: Optional[str] = Query(None),
        namespace: Optional[str] = Query(None),
        key_info: ScopedAPIKey = Depends(verify_scoped_auth(required_permissions))
    ) -> RequestContext:
        req_tenant = x_tenant or tenant
        req_namespace = x_namespace or namespace
        
        # Enforce key-level tenant/namespace constraints
        if key_info.tenant is not None:
            if req_tenant and req_tenant != key_info.tenant:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API Key restricted to a different tenant."
                )
            req_tenant = key_info.tenant
            
        if key_info.namespace is not None:
            if req_namespace and req_namespace != key_info.namespace:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API Key restricted to a different namespace."
                )
            req_namespace = key_info.namespace
            
        # Wildcard safety: "*" and "all" are query/read scopes for aggregating namespaces.
        # Write mutations must never store into a literal "*" partition directory; normalize to "default".
        if Permission.WRITE in required_permissions and req_namespace in ("*", "all"):
            req_namespace = os.getenv("DEFAULT_NAMESPACE") or "default"
            
        return RequestContext(tenant=req_tenant, namespace=req_namespace, key_info=key_info)
    return context_dependency



def get_shard_for_id(memory_id: str) -> Optional[str]:
    if not shard_nodes:
        return None
    if memory_id.startswith("shard"):
        parts = memory_id.split("-", 1)
        if len(parts) > 1:
            prefix = parts[0]  # e.g., "shard0"
            try:
                idx = int(prefix[5:])  # extract index
                if 0 <= idx < len(shard_nodes):
                    return shard_nodes[idx]
            except ValueError:
                pass
    return None

# -------------------------------------------------------------------------
# 3. Database State & Lifespan Management
# -------------------------------------------------------------------------
# Database pool map: (tenant, namespace) -> AsyncEpochDB instance (Shard Mode)
db_pool: Dict[tuple[Optional[str], Optional[str]], AsyncEpochDB] = {}
# Global reference for default database engine instance for backward compatibility
db: Optional[AsyncEpochDB] = None
# Global HTTP client session (Coordinator Mode)
client: Optional[httpx.AsyncClient] = None

# Locks for database initialization to prevent race conditions
db_init_locks: Dict[tuple[Optional[str], Optional[str]], asyncio.Lock] = {}
db_init_locks_lock = asyncio.Lock()

async def get_db_instance(tenant: Optional[str] = None, namespace: Optional[str] = None) -> AsyncEpochDB:
    key = (tenant, namespace)
    if key in db_pool:
        return db_pool[key]
        
    async with db_init_locks_lock:
        if key not in db_init_locks:
            db_init_locks[key] = asyncio.Lock()
        lock = db_init_locks[key]
        
    async with lock:
        if key in db_pool:
            return db_pool[key]
            
        storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
        logger.info(f"Initializing AsyncEpochDB for tenant={tenant}, namespace={namespace} under {storage_dir}")
        auto_extract = os.getenv("AUTO_EXTRACT", "true").lower() in ("1", "true", "yes")
        # Prefer explicit EPOCHDB_EXTRACTION_MODEL. Default "local" so a host
        # GEMINI_API_KEY / GEMINI_MODEL does not auto-select GoogleFactExtractor
        # (shards typically lack google-genai).
        extraction_model = (
            os.getenv("EPOCHDB_EXTRACTION_MODEL")
            or "local"
        )
        engine = AsyncEpochDB(
            storage_dir=storage_dir,
            embedding_model=EMBEDDING_MODEL,
            dim=EMBEDDING_DIM,
            wal_sync_interval=0.1,
            parquet_compression="zstd",
            parquet_compression_level=3,
            tenant=tenant,
            namespace=namespace,
            auto_extract=auto_extract,
            extraction_model=extraction_model,
        )
        await engine._get_db()
        
        # Warmup sequence
        try:
            await engine.query(text="system boot warmup", k=1)
        except Exception as e:
            if "dimension mismatch" in str(e).lower():
                logger.error(
                    f"Error warming up db instance {key}: {e}. "
                    f"Hint: Stored partition metadata/vectors were created with a different embedding dimension "
                    f"than the current model ({EMBEDDING_MODEL}, dim={EMBEDDING_DIM}). "
                    f"Either switch to a new namespace/tenant, or clear storage volumes (docker compose down -v) to reinitialize."
                )
            else:
                logger.error(f"Error warming up db instance {key}: {e}")
            
        db_pool[key] = engine
        return engine


def get_shard_db_stats(tenant: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, int]:
    """
    Fast, lock-free snapshot of shard database counts.
    Reads directly from in-memory hot/cold tier metadata structures without triggering
    SentenceTransformer embeddings, heavy write locks, or cold start delays.
    """
    instances = []
    if namespace in ("*", "all"):
        for (t, ns), inst in db_pool.items():
            if (t == tenant or (tenant is None and t is None)) and ns not in ("*", "all"):
                instances.append(inst)
        if tenant is None and db is not None:
            instances.append(db)
    elif tenant is not None or namespace is not None:
        if (tenant, namespace) in db_pool:
            instances.append(db_pool[(tenant, namespace)])
        elif db is not None:
            instances.append(db)
    else:
        seen = set()
        if db is not None:
            instances.append(db)
            seen.add(id(db))
        for inst in db_pool.values():
            if id(inst) not in seen:
                instances.append(inst)
                seen.add(id(inst))

    if instances:
        total_mem = 0
        total_l1 = 0
        total_l2 = 0
        total_ent = 0
        has_any = False
        for inst in instances:
            raw_db = getattr(inst, "_db", None)
            if raw_db is not None:
                try:
                    hot = getattr(raw_db, "hot_tier", None)
                    l1 = len(getattr(hot, "atoms", {})) if hot else 0
                    cold = getattr(raw_db, "cold_tier", None)
                    l2 = cold.get_total_atoms() if cold and hasattr(cold, "get_total_atoms") else 0
                    kg = getattr(raw_db, "kg_manager", None)
                    ent = len(kg.get_entities()) if kg and hasattr(kg, "get_entities") else 0
                    total_mem += (l1 + l2)
                    total_l1 += l1
                    total_l2 += l2
                    total_ent += ent
                    has_any = True
                except Exception:
                    pass
        if has_any:
            res = {
                "memory_count": total_mem,
                "l1_size": total_l1,
                "l2_size": total_l2,
                "entity_count": total_ent,
            }
            _LAST_SHARD_DB_STATS.update(res)
            return res

    return {
        "memory_count": _LAST_SHARD_DB_STATS.get("memory_count", 0),
        "l1_size": _LAST_SHARD_DB_STATS.get("l1_size", 0),
        "l2_size": _LAST_SHARD_DB_STATS.get("l2_size", 0),
        "entity_count": _LAST_SHARD_DB_STATS.get("entity_count", 0),
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifecycles of the ASGI application.
    """
    global db, client
    
    if NODE_MODE == "coordinator":
        logger.info("Initializing Sharded Clustering Coordinator Gateway...")
        if not shard_nodes:
            logger.warning("No SHARD_NODES configured for coordinator. Check env config.")
        headers = {}
        if INTERNAL_AUTH_TOKEN:
            headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
        logger.info(
            "Coordinator HTTP client timeout=%.1fs (COORDINATOR_HTTP_TIMEOUT)",
            COORDINATOR_HTTP_TIMEOUT,
        )
        client = httpx.AsyncClient(headers=headers, timeout=COORDINATOR_HTTP_TIMEOUT)
        polling_task = asyncio.create_task(poll_shards_loop())
        yield
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
        await client.aclose()
        logger.info("Coordinator HTTP client session closed cleanly.")
    else:
        logger.info("Initializing default AsyncEpochDB engine in Shard Mode...")
        try:
            # Pre-initialize and warm up default (root) database
            db = await get_db_instance(None, None)
            # Also warm the app tenant partition when configured (e.g. Percepta
            # global/default) so the first request does not cold-load MiniLM again.
            if DEFAULT_TENANT is not None or DEFAULT_NAMESPACE is not None:
                logger.info(
                    "Pre-warming app partition tenant=%s namespace=%s",
                    DEFAULT_TENANT,
                    DEFAULT_NAMESPACE,
                )
                await get_db_instance(DEFAULT_TENANT, DEFAULT_NAMESPACE)
            logger.info("Default db warmed up. AsyncEpochDB database pool ready.")
            yield
        except Exception as e:
            logger.critical(f"Fatal error during engine startup sequence: {str(e)}")
            raise e
        finally:
            logger.info("Closing all active database instances in pool...")
            for key, engine in list(db_pool.items()):
                try:
                    await engine.close()
                except Exception as e:
                    logger.error(f"Failed to close engine instance {key}: {e}")
            db_pool.clear()
            logger.info("Database pool context exited cleanly.")
            logger.info("Database context exited cleanly. All resources released.")

# Initialize FastAPI application with lifespan management
app = FastAPI(
    title="EpochDB Core Server",
    description="Asynchronous high-concurrency memory engine interface.",
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------------------------------------------------
# 4. Data Transfer Objects (Pydantic Models)
# -------------------------------------------------------------------------
class MemoryPayload(BaseModel):
    text: Optional[str] = Field(default=None, description="The factual memory text to write to the engine.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional structural metadata or graph triples.")
    id: Optional[str] = Field(default=None, description="Optional unique identifier (pre-calculated or forwarded).")
    memory_type: Optional[str] = Field(default=None, description="Optional memory type: 'general', 'episodic', 'profile', 'working', or 'skill'.")
    triples: Optional[Any] = Field(default=None, description="Optional structural graph triples.")
    subject: Optional[str] = Field(default=None, description="Optional single triplet subject.")
    predicate: Optional[str] = Field(default=None, description="Optional single triplet predicate.")
    object: Optional[str] = Field(default=None, description="Optional single triplet object.")

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            subj = data.get("subject")
            pred = data.get("predicate")
            obj = data.get("object")
            triples = data.get("triples")
            meta = data.get("metadata") or {}
            meta_triples = meta.get("triples") if isinstance(meta, dict) else None

            collected = []
            if subj and pred and obj:
                collected.append([str(subj).strip(), str(pred).strip(), str(obj).strip()])
            if triples and isinstance(triples, (list, tuple)):
                for t in triples:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        collected.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])
            if meta_triples and isinstance(meta_triples, (list, tuple)):
                for t in meta_triples:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        collected.append([str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()])

            if collected:
                data["triples"] = collected
                if not data.get("text"):
                    # Auto-generate human-readable memory sentence from the triples
                    data["text"] = ". ".join([f"{t[0]} {t[1].replace('_', ' ')} {t[2]}" for t in collected]) + "."

            if not data.get("text") and not collected:
                raise ValueError("Either 'text', 'triples', or subject/predicate/object must be provided.")
        return data

class BatchMemoryPayload(BaseModel):
    items: List[Any] = Field(..., description="List of memory items (text strings or dict objects) to ingest in batch.")

class ReplacePayload(BaseModel):
    atom_id: str = Field(..., description="Existing atom ID to replace in-place.")
    text: str = Field(..., description="New memory payload text.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata dictionary.")
    triples: Optional[Any] = Field(default=None, description="Optional updated triples.")

class QueryPayload(BaseModel):
    query: Optional[str] = Field(default=None, description="The semantic search query string.")
    text: Optional[str] = Field(default=None, description="Alias for query text for client parity.")
    k: int = Field(default=1, ge=1, le=100, description="The number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="MongoDB-style metadata filter evaluation parameters.")
    memory_type: Optional[str] = Field(default=None, description="Optional filter by memory type: 'general', 'episodic', 'profile', 'working', or 'skill'.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns to retrieve around the matched memory.")
    expand_hops: int = Field(default=0, ge=0, le=10, description="Relational KG expansion hops during retrieval.")
    min_score: float = Field(default=0.0, description="Minimum similarity score threshold.")
    cold_search_mode: Optional[str] = Field(default=None, description="Cold tier search mode: 'probe' or 'exhaustive'.")

    def get_query_text(self) -> str:
        return self.query or self.text or ""

class MultiHopPayload(BaseModel):
    query: Optional[str] = Field(default=None, description="The query string.")
    text: Optional[str] = Field(default=None, description="Alias for query text.")
    hops: int = Field(default=2, ge=1, le=10, description="Number of relational KG expansion hops.")
    k: int = Field(default=5, ge=1, le=100, description="Number of candidate matches to return.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns.")
    cold_search_mode: Optional[str] = Field(default=None, description="Cold tier search mode: 'probe' or 'exhaustive'.")

    def get_query_text(self) -> str:
        return self.query or self.text or ""

class AdaptiveQueryPayload(BaseModel):
    query: str = Field(..., description="The natural language search query.")
    k: int = Field(default=5, ge=1, le=100, description="The number of candidate matches to return.")
    context_window: int = Field(default=0, ge=0, description="Number of chronological context turns to retrieve around the matched memory.")

class GetPayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to retrieve.")

class UpdatePayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to update.")
    text: Optional[str] = Field(default=None, description="Optional new text payload.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional new metadata dictionary.")
    triples: Optional[Any] = Field(default=None, description="Optional updated triples.")

class DeletePayload(BaseModel):
    memory_id: str = Field(..., description="The unique identifier of the memory to delete.")
    hard: bool = Field(default=False, description="Whether to hard delete or soft delete.")

class TimelinePayload(BaseModel):
    entity_id: Optional[str] = Field(default=None, description="Optional entity ID.")
    start: Optional[float] = Field(default=None, description="Optional start timestamp.")
    end: Optional[float] = Field(default=None, description="Optional end timestamp.")

class ExportPayload(BaseModel):
    data_type: str = Field(default="all", description="Export category: 'all', 'skills', 'profiles', 'triplets', or 'memories'.")
    memory_type: Optional[str] = Field(default=None, description="Optional memory type filter when exporting memories.")
    tenant: Optional[str] = Field(default=None, description="Optional tenant.")
    namespace: Optional[str] = Field(default=None, description="Optional namespace.")

class ImportPayload(BaseModel):
    data: Dict[str, Any] = Field(..., description="Export bundle dictionary containing skills, profiles, memories, or triplets.")
    data_type: Optional[str] = Field(default=None, description="Optional category filter to import from the bundle.")
    mode: str = Field(default="merge", description="Import mode: 'merge' or 'replace'.")

class SkillCreatePayload(BaseModel):
    skill_name: str = Field(..., description="Unique human-readable skill name.")
    description: str = Field(..., description="Detailed description of what the procedural skill accomplishes.")
    steps: List[Union[Dict[str, Any], str, Any]] = Field(default_factory=list, description="Ordered procedural steps.")
    tool_schema: Optional[Dict[str, Any]] = Field(default=None, description="Optional JSON Schema tool/function definition.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata.")
    skill_id: Optional[str] = Field(default=None, description="Optional stable unique identifier.")
    decision_rules: Optional[List[str]] = Field(default=None, description="Optional decision rules.")
    triples: Optional[Any] = Field(default=None, description="Optional structural knowledge graph triples.")

class SkillGetPayload(BaseModel):
    skill_name: str = Field(..., description="Unique skill name or identifier.")

class ProfileCreatePayload(BaseModel):
    user_id: str = Field(..., description="Unique user identifier.")
    fact_text: str = Field(..., description="Factual profile or preference text.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata.")

class ProfileGetPayload(BaseModel):
    user_id: str = Field(..., description="Unique user identifier.")

class SnapshotPayload(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID filter for summary snapshot.")

class ProposePayload(BaseModel):
    text: str = Field(..., description="Factual memory text to propose for symbolic validation.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional structural metadata.")
    triples: Optional[Any] = Field(default=None, description="Optional graph triples.")
    id: Optional[str] = Field(default=None, description="Optional predefined identifier.")

# -------------------------------------------------------------------------
# 5. API Core Router Endpoints
# -------------------------------------------------------------------------
@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """
    Liveness and Readiness probe endpoint.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator gateway not initialized.")
        if shard_nodes:
            # Prefer the background poll cache. Live fan-out with a short timeout
            # fails whenever a single-worker shard is mid-embed (CPU-bound MiniLM),
            # which flaps Docker healthchecks despite shards being up.
            # Match poll_shards_loop: only ConnectError means offline; ReadTimeout
            # means busy/overload and must not 503 the coordinator.
            uncached = [s for s in shard_nodes if s not in SHARD_METRICS_CACHE]
            if uncached:
                probe_timeout = float(os.getenv("COORDINATOR_HEALTHZ_TIMEOUT", "10"))
                tasks = [client.get(f"{shard}/healthz", timeout=probe_timeout) for shard in uncached]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for shard_url, resp in zip(uncached, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        continue
                    err = format_shard_error(resp)
                    # Hard down (DNS/connect) or explicit non-200 → not ready.
                    # ReadTimeout / overload → busy shard; do not 503 the gateway.
                    if should_mark_shard_offline(resp) or isinstance(resp, httpx.Response):
                        logger.warning(f"Shard health probe failed: {shard_url} -> {err}")
                        raise HTTPException(
                            status_code=503,
                            detail="One or more backend shards are unhealthy or warming up.",
                        )
                    logger.warning(f"Shard health probe deferred (busy/warming): {shard_url} -> {err}")
            else:
                offline = [
                    s for s in shard_nodes
                    if SHARD_METRICS_CACHE.get(s, {}).get("status") == "unhealthy"
                ]
                if offline:
                    logger.warning(f"Coordinator healthz: cached unhealthy shards={offline}")
                    raise HTTPException(
                        status_code=503,
                        detail="One or more backend shards are unhealthy or warming up.",
                    )
        return {"status": "healthy", "mode": "coordinator", "version": SERVER_VERSION}
    else:
        global db
        if db is None:
            try:
                db = await get_db_instance(None, None)
            except Exception as e:
                logger.error(f"Failed to initialize storage engine in healthz: {e}")
        if db is None:
            raise HTTPException(status_code=503, detail="Storage engine not ready.")
        return {"status": "healthy", "mode": "shard", "version": SERVER_VERSION}

# Helper to build headers when coordinator forwards to shards
def get_forward_headers(ctx: RequestContext) -> Dict[str, str]:
    headers = {}
    if ctx.tenant:
        headers["X-Tenant"] = ctx.tenant
    if ctx.namespace:
        headers["X-Namespace"] = ctx.namespace
    if INTERNAL_AUTH_TOKEN:
        headers["X-Internal-Token"] = INTERNAL_AUTH_TOKEN
    return headers


def _scan_local_databases() -> List[Dict[str, Optional[str]]]:
    storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
    db_keys = set()
    for key in db_pool.keys():
        if key[1] not in ("*", "all"):
            db_keys.add(key)
    if db is not None:
        db_keys.add((None, None))
    if os.path.exists(os.path.join(storage_dir, "metadata.json")):
        db_keys.add((None, None))
    ns_dir = os.path.join(storage_dir, "ns")
    if os.path.exists(ns_dir):
        try:
            for namespace in os.listdir(ns_dir):
                if namespace not in ("*", "all") and os.path.isdir(os.path.join(ns_dir, namespace)):
                    db_keys.add((None, namespace))
        except Exception as e:
            logger.error(f"Error listing namespace directory: {e}")
    tenants_dir = os.path.join(storage_dir, "tenants")
    if os.path.exists(tenants_dir):
        try:
            for tenant in os.listdir(tenants_dir):
                tenant_path = os.path.join(tenants_dir, tenant)
                if os.path.isdir(tenant_path):
                    if os.path.exists(os.path.join(tenant_path, "metadata.json")):
                        db_keys.add((tenant, None))
                    tenant_ns_dir = os.path.join(tenant_path, "ns")
                    if os.path.exists(tenant_ns_dir):
                        for namespace in os.listdir(tenant_ns_dir):
                            if namespace not in ("*", "all") and os.path.isdir(os.path.join(tenant_ns_dir, namespace)):
                                db_keys.add((tenant, namespace))
        except Exception as e:
            logger.error(f"Error listing tenants directory: {e}")
    return [{"tenant": t, "namespace": ns} for t, ns in sorted(db_keys, key=lambda x: (x[0] or "", x[1] or ""))]


def get_namespaces_for_tenant(tenant: Optional[str]) -> List[Optional[str]]:
    all_dbs = _scan_local_databases()
    namespaces = set()
    for item in all_dbs:
        t = item["tenant"]
        ns = item["namespace"]
        if t == tenant or (tenant is None and t is None) or (tenant == "global" and t in (None, "global")):
            if ns not in ("*", "all"):
                namespaces.add(ns)
    return sorted(list(namespaces), key=lambda x: (x is not None, x or ""))


# Shard database partitions administration schemas and routes
class ResetPayload(BaseModel):
    tenant: Optional[str] = None
    namespace: Optional[str] = None

@app.get("/admin/databases")
async def admin_list_databases(ctx: RequestContext = Depends(get_context({Permission.ADMIN}))):
    """
    Lists all local database partitions (tenants and namespaces) on this shard.
    Only supported in Shard Mode.
    """
    if NODE_MODE == "coordinator":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supported in Shard Mode.")
    return {"databases": _scan_local_databases()}


@app.get("/databases")
async def list_databases(ctx: RequestContext = Depends(get_context({Permission.READ}))):
    """
    Lists all available database partitions (tenants and namespaces) for visualization scope autocomplete.
    """
    if NODE_MODE == "coordinator":
        if not client or not shard_nodes:
            return {"databases": []}
        target_shard = get_healthy_replica_for_group(shard_groups[0]) if shard_groups else shard_nodes[0]
        try:
            resp = await client.get(f"{target_shard}/databases", headers=get_forward_headers(ctx))
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Error querying databases from shard: {e}")
        return {"databases": []}
    else:
        return {"databases": _scan_local_databases()}


@app.post("/admin/toggle_sync")
async def toggle_sync(
    enabled: bool,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    """
    Toggles coordinator's background health check shard data synchronization loop.
    Only supported in Coordinator Mode.
    """
    if NODE_MODE != "coordinator":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supported in Coordinator Mode.")
    
    global BACKGROUND_SYNC_ENABLED
    BACKGROUND_SYNC_ENABLED = enabled
    logger.info(f"Background shard synchronization set to {enabled}")
    return {"status": "success", "enabled": BACKGROUND_SYNC_ENABLED}


@app.post("/admin/reset")
async def admin_reset_database(
    payload: ResetPayload,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    """
    Resets and completely deletes a database partition (tenant and namespace) on this shard.
    Only supported in Shard Mode.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        
        all_replica_nodes = []
        for group in shard_groups:
            all_replica_nodes.extend(group)
            
        tasks = [
            client.post(
                f"{node}/admin/reset",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in all_replica_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = []
        for node, resp in zip(all_replica_nodes, responses):
            if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                logger.error(f"Reset failed on node {node}: {err_msg}")
                errors.append(f"{node}: {err_msg}")
                
        if errors:
            raise HTTPException(status_code=500, detail=f"Failed to reset partition on all replica nodes: {', '.join(errors)}")
            
        invalidate_cache(payload.tenant, payload.namespace)
        return {"status": "success", "message": f"Cluster-wide database partition reset successfully."}
        
    key = (payload.tenant, payload.namespace)
    logger.info(f"Admin reset request received for tenant={payload.tenant}, namespace={payload.namespace}")
    
    # 1. Close and remove the database instance from pool
    if key in db_pool:
        db_instance = db_pool[key]
        try:
            await db_instance.close()
        except Exception as e:
            logger.error(f"Error closing db instance for reset: {e}")
        del db_pool[key]
        
    # 2. Delete storage files
    storage_dir = os.getenv("STORAGE_DIR", "./shared_memory")
    if payload.tenant:
        storage_dir = os.path.join(storage_dir, "tenants", payload.tenant)
    if payload.namespace:
        storage_dir = os.path.join(storage_dir, "ns", payload.namespace)
    storage_dir = os.path.abspath(storage_dir)
    
    if os.path.exists(storage_dir):
        try:
            import shutil
            shutil.rmtree(storage_dir)
            logger.info(f"Deleted storage directory {storage_dir}")
        except Exception as e:
            logger.error(f"Failed to delete storage directory {storage_dir}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear partition storage: {str(e)}"
            )
            
    global db
    if payload.tenant is None and payload.namespace is None:
        db = await get_db_instance(None, None)
        
    invalidate_cache(payload.tenant, payload.namespace)
    return {"status": "success", "message": f"Database partition reset successfully."}


# Key administration schemas
class KeyCreatePayload(BaseModel):
    permissions: List[str]
    tenant: Optional[str] = None
    namespace: Optional[str] = None
    expires_at: Optional[float] = None
    description: Optional[str] = None

# Key administration routes
@app.post("/admin/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: KeyCreatePayload,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    key_id, raw_key = keystore.create_key(
        permissions=payload.permissions,
        tenant=payload.tenant,
        namespace=payload.namespace,
        expires_at=payload.expires_at,
        description=payload.description
    )
    return {"key_id": key_id, "api_key": raw_key}

@app.delete("/admin/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    success = keystore.revoke_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"status": "success", "message": f"Key {key_id} revoked."}

@app.get("/admin/keys")
async def list_api_keys(
    ctx: RequestContext = Depends(get_context({Permission.ADMIN}))
):
    keystore = get_keystore()
    keys = keystore.list_keys()
    return {"keys": [k.model_dump() for k in keys]}


@app.post("/remember", status_code=status.HTTP_201_CREATED)
async def remember(payload: MemoryPayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.WRITE}))):
    """
    Appends a new memory atom to the Hot Tier (RAM) and schedules background WAL logging.
    In coordinator mode, routes writes using consistent hashing and ID prefixing.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available to route write request.")
        
        original_predefined_id = payload.id
        if original_predefined_id:
            target_shard = get_shard_for_id(original_predefined_id)
            if target_shard:
                atom_id = original_predefined_id
            else:
                target_shard = get_healthy_shard(payload.text)
                shard_idx = shard_nodes.index(target_shard)
                atom_id = f"shard{shard_idx}-{original_predefined_id}"
        else:
            target_shard = get_healthy_shard(payload.text)
            shard_idx = shard_nodes.index(target_shard)
            atom_id = f"shard{shard_idx}-{uuid.uuid4().hex}"

        max_attempts = max(1, int(os.getenv("COORDINATOR_WRITE_RETRIES", "3")))
        for attempt in range(1, max_attempts + 1):
            
            try:
                target_payload = {
                    "text": payload.text,
                    "metadata": payload.metadata,
                    "id": atom_id,
                    "memory_type": payload.memory_type,
                    "triples": payload.triples
                }
                group = get_replica_group_for_node(target_shard)
                tasks = [
                    client.post(
                        f"{node}/remember",
                        json=target_payload,
                        headers=get_forward_headers(ctx)
                    ) for node in group
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_nodes = []
                failures = []
                for node, resp in zip(group, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 201:
                        success_nodes.append(node)
                    else:
                        err_msg = format_shard_error(resp)
                        logger.warning(f"Replication write failed to node {node}: {err_msg}")
                        failures.append((node, err_msg))
                        if should_mark_shard_offline(resp):
                            SHARD_METRICS_CACHE[node] = {
                                "status": "offline",
                                "cpu": 0.0,
                                "ram": {"percent": 0.0},
                                "disk": {"percent": 0.0}
                            }
                
                consistency = get_consistency_level(request)
                n = len(group)
                if consistency == ConsistencyLevel.ALL:
                    required_success = n
                elif consistency == ConsistencyLevel.QUORUM:
                    required_success = n // 2 + 1
                else:
                    required_success = 1
                
                if len(success_nodes) >= required_success:
                    invalidate_cache(ctx.tenant, ctx.namespace)
                    await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                    return {"status": "success", "id": atom_id}
                else:
                    last_err = failures[-1][1] if failures else "Unknown error"
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"Owning shard group could not accept write. Required: "
                            f"{consistency.value} ({required_success}), got: "
                            f"{len(success_nodes)} success(es). Last error: {last_err}"
                        ),
                        headers={"Retry-After": "5"},
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to forward write to shard set {target_shard} on attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Owning shard group could not accept write: {str(e)}",
                        headers={"Retry-After": "5"},
                    )
            
    else:
        # Shard Mode (Storage Node)
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            if payload.metadata is None:
                payload.metadata = {}
            if "_updated_at" not in payload.metadata:
                payload.metadata["_updated_at"] = time.time()
                
            provided_triples = list(payload.triples or []) or list(payload.metadata.get("triples") or [])
            if provided_triples and "triples" not in payload.metadata:
                payload.metadata["triples"] = provided_triples

            if payload.id:
                engine = await active_db._get_db()
                text = payload.text
                metadata = payload.metadata
                
                # Manual embedding generation to support predefined atom_id
                if engine._model_name:
                    embedder = engine._get_embedder()
                    emb = await run_embedding(
                        embedder.encode,
                        text,
                        is_query=False,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    embedding = np.array(emb, dtype=np.float32)
                    if len(embedding) > engine.dim:
                        embedding = embedding[:engine.dim]
                        norm = np.linalg.norm(embedding)
                        if norm > 1e-10:
                            embedding /= norm
                else:
                    embedding = np.zeros(engine.dim, dtype=np.float32)
                    
                provided_triples = metadata.get("triples") or payload.triples or []

                def _extract_triples_for_ingest() -> list:
                    from epochdb.core.fact_extractor import FactExtractor

                    if getattr(engine, "auto_extract", False):
                        if engine._fact_extractor is None:
                            engine._fact_extractor = FactExtractor(
                                engine, engine.extraction_model
                            )
                        return engine._fact_extractor.extract(text)

                    def _pair_triples(items: list) -> list:
                        seen = set()
                        entities = []
                        for item in items:
                            s = str(item)
                            if s and s not in seen:
                                seen.add(s)
                                entities.append(s)
                        if len(entities) >= 2:
                            res = []
                            for i in range(len(entities) - 1):
                                res.append((entities[i], "co_occurs_with", entities[i + 1]))
                            if len(entities) > 2:
                                res.append((entities[0], "co_occurs_with", entities[-1]))
                            return res
                        elif len(entities) == 1:
                            return [(entities[0], "mentions", entities[0])]
                        return []

                    extracted = engine.extract_entities(text)
                    if extracted:
                        t_lower = text.lower()
                        stopwords = {
                            "for", "the", "and", "with", "from", "into", "that", "this",
                            "some", "all", "are", "was", "were", "been", "has", "had",
                            "have", "not", "but", "about", "above", "after", "again", "out"
                        }
                        valid_extracted = []
                        for ent in extracted:
                            ent_s = str(ent).strip()
                            ent_l = ent_s.lower()
                            if ent_l in t_lower:
                                valid_extracted.append(ent_s)
                            elif " " in ent_s:
                                parts = [p for p in ent_l.replace("-", " ").split() if len(p) > 2 and p not in stopwords]
                                if parts:
                                    matched_count = sum(1 for p in parts if p in t_lower)
                                    if (len(parts) <= 2 and matched_count == len(parts)) or (matched_count >= 2 and (matched_count / len(parts)) >= 0.5):
                                        valid_extracted.append(ent_s)
                            else:
                                if ent_l not in stopwords and ent_l in t_lower:
                                    valid_extracted.append(ent_s)
                        if valid_extracted:
                            res = _pair_triples(valid_extracted)
                            if res:
                                return res

                    # Match LocalFactExtractor fallback for brand-new knowledge graphs.
                    words = [w.strip(".,!?;:()\"'") for w in text.split() if w.strip()]
                    nouns = [w for w in words if w and w[0].isupper()]
                    if not nouns:
                        nouns = [w for w in words if len(w) > 3][:3]
                    return _pair_triples(nouns)

                # Always discover entity links when AUTO_EXTRACT is on, and merge
                # them with any caller-provided structural triples (rebuild/index).
                if getattr(engine, "auto_extract", False) or not provided_triples:
                    discovered = await asyncio.to_thread(_extract_triples_for_ingest)
                else:
                    discovered = []

                merged = []
                seen_triples = set()
                for t in list(provided_triples) + list(discovered or []):
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        key = (str(t[0]), str(t[1]), str(t[2]))
                    else:
                        continue
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    merged.append(list(key))
                triples = merged
                metadata["triples"] = triples

                # If the atom already exists (hot or cold), replace triples in-place.
                existing = payload.id in engine.hot_tier.atoms
                if not existing:
                    for epoch_id in engine.cold_tier.get_all_epochs():
                        if engine.cold_tier.load_atom_metadata(epoch_id, [payload.id]):
                            existing = True
                            break

                if existing:
                    atom_id = await asyncio.to_thread(
                        engine.replace_memory,
                        payload.id,
                        text,
                        embedding,
                        [tuple(t) for t in triples],
                        metadata,
                    )
                    logger.info(
                        f"Replaced triples for atom {atom_id}: '{text[:40]}...'"
                    )
                    invalidate_cache(ctx.tenant, ctx.namespace)
                    await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                    return {"status": "success", "id": atom_id}

                atom_id = await asyncio.to_thread(
                    engine.add_memory,
                    payload=text,
                    embedding=embedding,
                    triples=triples,
                    metadata=metadata,
                    atom_id=payload.id
                )
                
                # Set memory_type on the atom if specified
                if payload.memory_type:
                    from epochdb.core.atom import MemoryType
                    try:
                        mt = MemoryType(payload.memory_type)
                        atom = engine.hot_tier.atoms.get(atom_id)
                        if atom:
                            atom.memory_type = mt
                    except ValueError:
                        pass
                
                logger.info(f"Ingested atom with fixed ID {atom_id}: '{text[:40]}...'")
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                return {"status": "success", "id": atom_id}
            else:
                async with embedding_slot():
                    atom_id = await active_db.remember(
                        text=payload.text,
                        triples=provided_triples or None,
                        metadata=payload.metadata,
                        memory_type=payload.memory_type,
                    )
                logger.info(f"Ingested atom: '{payload.text[:40]}...'")
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "write", "id": atom_id})
                return {"status": "success", "id": atom_id}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to commit memory write block: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal storage layer mutation rejected: {str(e)}")

@app.post("/remember_batch", status_code=status.HTTP_201_CREATED)
async def remember_batch(
    payload: BatchMemoryPayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Bulk ingestion endpoint for adding multiple memory atoms.
    Coordinator partitions items by target shard and routes in parallel batches.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available to route batch request.")

        grouped_items: Dict[str, List[Dict[str, Any]]] = {}
        all_ids: List[str] = []

        for item in payload.items:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {"text": str(item)})
            text = item_dict.get("text") or item_dict.get("payload") or ""
            target_id = item_dict.get("id") or item_dict.get("atom_id")

            if target_id:
                target_shard = get_shard_for_id(target_id)
                if not target_shard:
                    target_shard = get_healthy_shard(text)
                    shard_idx = shard_nodes.index(target_shard)
                    target_id = f"shard{shard_idx}-{target_id}"
            else:
                target_shard = get_healthy_shard(text)
                shard_idx = shard_nodes.index(target_shard)
                target_id = f"shard{shard_idx}-{uuid.uuid4().hex}"

            item_dict["id"] = target_id
            item_dict["atom_id"] = target_id
            all_ids.append(target_id)

            if target_shard not in grouped_items:
                grouped_items[target_shard] = []
            grouped_items[target_shard].append(item_dict)

        forward_headers = get_forward_headers(ctx)
        consistency = get_consistency_level(request)

        async def send_group_batch(primary_node: str, items_for_shard: List[Dict[str, Any]]):
            group = get_replica_group_for_node(primary_node)
            tasks = [
                client.post(
                    f"{node}/remember_batch",
                    json={"items": items_for_shard},
                    headers=forward_headers
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = 0
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 201:
                    success_count += 1
                else:
                    err_msg = format_shard_error(resp)
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }

            n = len(group)
            required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)
            if success_count < required:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=503,
                    detail=f"Batch replication failed for shard group {primary_node}. Required: {consistency.value} ({required}), got: {success_count}. Last error: {last_err}",
                    headers={"Retry-After": "5"}
                )

        await asyncio.gather(*[send_group_batch(shard, items) for shard, items in grouped_items.items()])

        invalidate_cache(ctx.tenant, ctx.namespace)
        await broadcast_sse_event("update", {"type": "batch_write", "count": len(all_ids)})
        return {"status": "success", "atom_ids": all_ids, "count": len(all_ids)}

    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        formatted_items = []
        now_ts = time.time()
        for item in payload.items:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {"text": str(item)})
            text = item_dict.get("text") or item_dict.get("payload") or ""
            metadata = dict(item_dict.get("metadata") or {})
            if "_updated_at" not in metadata:
                metadata["_updated_at"] = now_ts
            if "triples" not in metadata and item_dict.get("triples"):
                metadata["triples"] = item_dict["triples"]
            atom_id = item_dict.get("id") or item_dict.get("atom_id")
            memory_type = item_dict.get("memory_type")

            formatted_items.append({
                "text": text,
                "metadata": metadata,
                "id": atom_id,
                "memory_type": memory_type,
            })

        async with embedding_slot():
            atom_ids = []
            for itm in formatted_items:
                aid = await active_db.remember(
                    text=itm["text"],
                    metadata=itm["metadata"],
                    memory_type=itm["memory_type"],
                    atom_id=itm["id"],
                )
                atom_ids.append(aid)

        invalidate_cache(ctx.tenant, ctx.namespace)
        await broadcast_sse_event("update", {"type": "batch_write", "count": len(atom_ids)})
        return {"status": "success", "atom_ids": atom_ids, "count": len(atom_ids)}

@app.post("/replace")
async def replace_memory(
    payload: ReplacePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Performs durable in-place replacement of an existing memory atom.
    Replaces payload, embedding, and knowledge graph triples without creating duplicate versions.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

        target_shard = get_shard_for_id(payload.atom_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/replace",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }

            consistency = get_consistency_level(request)
            n = len(group)
            required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)
            if len(success_nodes) >= required:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "replace", "id": payload.atom_id})
                return {"status": "success", "id": payload.atom_id}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Replace consistency check failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/replace",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "replace", "id": payload.atom_id})
            return {"status": "success", "id": payload.atom_id}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            engine = await active_db._get_db()
            meta = dict(payload.metadata or {})
            meta["_updated_at"] = time.time()
            if payload.triples is not None:
                meta["triples"] = payload.triples

            if engine._model_name:
                embedder = engine._get_embedder()
                emb = await run_embedding(
                    embedder.encode,
                    payload.text,
                    is_query=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                embedding = np.array(emb, dtype=np.float32)
                if len(embedding) > engine.dim:
                    embedding = embedding[:engine.dim]
                    norm = np.linalg.norm(embedding)
                    if norm > 1e-10:
                        embedding /= norm
            else:
                embedding = np.zeros(engine.dim, dtype=np.float32)

            triples = meta.get("triples") or []
            atom_id = await asyncio.to_thread(
                engine.replace_memory,
                payload.atom_id,
                payload.text,
                embedding,
                [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) >= 3],
                meta
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "replace", "id": atom_id})
            return {"status": "success", "id": atom_id}
        except Exception as e:
            logger.error(f"Error replacing memory {payload.atom_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/propose", status_code=status.HTTP_200_OK)
async def propose_memory(
    payload: ProposePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Two-phase state commit (Neuro-Symbolic LCAG verification).
    Stages the memory, executes configured symbolic invariants, and atomically commits or discards.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        target_shard = get_healthy_shard(payload.text)
        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/propose",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        approved_result = None
        rejections = []
        failures = []

        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "APPROVED":
                    success_nodes.append(node)
                    approved_result = data
                else:
                    rejections.append((node, data))
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {
                        "status": "offline",
                        "cpu": 0.0,
                        "ram": {"percent": 0.0},
                        "disk": {"percent": 0.0}
                    }

        if rejections and not success_nodes:
            return rejections[0][1]

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required and approved_result:
            invalidate_cache(ctx.tenant, ctx.namespace)
            mid = approved_result.get("memory_id")
            await broadcast_sse_event("update", {"type": "propose_approved", "id": mid})
            return approved_result
        else:
            last_err = failures[-1][1] if failures else (rejections[-1][1].get("error") if rejections else "Unknown validation failure")
            raise HTTPException(
                status_code=503,
                detail=f"Proposal replication failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            res = await active_db.propose(
                text=payload.text,
                metadata=payload.metadata,
                triples=payload.triples,
            )
            if res.get("status") == "APPROVED":
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "propose_approved", "id": res.get("memory_id")})
            return res
        except Exception as e:
            logger.error(f"Error in propose memory validation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/get")
async def get_memory(
    payload: GetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a specific memory by its unique ID.
    Coordinator routes directly if ID is prefixed, otherwise broadcasts.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_shard = get_shard_for_id(payload.memory_id)
            if target_shard:
                group = get_replica_group_for_node(target_shard)
                tasks = [
                    client.post(
                        f"{node}/get",
                        json=payload.model_dump(),
                        headers=get_forward_headers(ctx)
                    ) for node in group
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                valid_replies = []
                failures = []
                for node, resp in zip(group, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        valid_replies.append((node, resp.json()))
                    else:
                        err_msg = str(resp) if isinstance(resp, Exception) else f"Status code {resp.status_code if resp else 'unknown'}"
                        failures.append((node, err_msg))
                        
                n = len(group)
                consistency = get_consistency_level(request)
                if consistency == ConsistencyLevel.ALL:
                    required_success = n
                elif consistency == ConsistencyLevel.QUORUM:
                    required_success = n // 2 + 1
                else:
                    required_success = 1
                    
                if len(valid_replies) < required_success:
                    last_err = failures[-1][1] if failures else "Unknown read failure"
                    raise HTTPException(
                        status_code=500,
                        detail=f"Read consistency check failed. Required: {consistency.value} ({required_success}), got: {len(valid_replies)} valid replies. Last error: {last_err}"
                    )
                    
                memories_found = [(node, data) for node, data in valid_replies if data and "id" in data]
                
                if not memories_found:
                    return {}
                    
                # A memory is newer if it has a higher _updated_at in metadata, or fallback to created_at
                def get_timestamp(data):
                    meta = data.get("metadata") or {}
                    return meta.get("_updated_at") or data.get("created_at") or 0.0
                    
                latest_node, latest_data = max(memories_found, key=lambda x: get_timestamp(x[1]))
                latest_ts = get_timestamp(latest_data)
                
                stale_replicas = []
                for node, data in valid_replies:
                    if not data or "id" not in data:
                        stale_replicas.append(node)
                    elif get_timestamp(data) < latest_ts:
                        stale_replicas.append(node)
                        
                if stale_replicas:
                    logger.info(f"Read Repair triggered: repairing stale/missing memory {payload.memory_id} on replicas: {stale_replicas}")
                    async def run_read_repair(target_nodes=stale_replicas, memory_data=latest_data):
                        repair_payload = {
                            "text": memory_data.get("payload") or memory_data.get("text"),
                            "metadata": memory_data.get("metadata", {}),
                            "id": memory_data.get("id"),
                            "memory_type": memory_data.get("memory_type")
                        }
                        headers = get_forward_headers(ctx)
                        for node in target_nodes:
                            try:
                                res = await client.post(f"{node}/remember", json=repair_payload, headers=headers)
                                if res.status_code == 201:
                                    logger.info(f"Successfully repaired memory {memory_data.get('id')} on node {node}")
                                else:
                                    logger.error(f"Failed to repair memory on node {node}: status {res.status_code}")
                            except Exception as re_err:
                                logger.error(f"Read repair request failed for node {node}: {re_err}")
                                
                    asyncio.create_task(run_read_repair())
                    
                return latest_data
            else:
                # Broadcast to all shards in parallel (one healthy replica per shard group)
                target_nodes = []
                for group in shard_groups:
                    node = get_healthy_replica_for_group(group)
                    if node:
                        target_nodes.append(node)
                tasks = [
                    client.post(
                        f"{node}/get",
                        json=payload.model_dump(),
                        headers=get_forward_headers(ctx)
                    ) for node in target_nodes
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for resp in responses:
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        if data and "id" in data:
                            return data
                return {}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mem = await active_db.get(payload.memory_id)
                if mem:
                    return mem._atom.to_dict()
                return {}
            except Exception as e:
                logger.error(f"Error resolving memory retrieval: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    consistency = get_consistency_level(request)
    if consistency in (ConsistencyLevel.QUORUM, ConsistencyLevel.ALL):
        return await fetch()
    else:
        return await handle_cached_read("get", payload, request, response, ctx, fetch)

@app.post("/query")
async def query_memories(
    payload: QueryPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Performs semantic and hybrid search across the memory database.
    Coordinator parallelizes requests to all shards and merges/re-ranks results.
    """
    query_text = payload.get_query_text()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            forward_data = payload.model_dump()
            forward_data["query"] = query_text
            forward_data["text"] = query_text

            tasks = [
                client.post(
                    f"{node}/query",
                    json=forward_data,
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            failed_shards = 0
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    all_results.extend(data.get("results") or data.get("memories") or [])
                else:
                    failed_shards += 1
                    logger.error(f"Error querying shard {node}: {format_shard_error(resp)}")
                    
            if payload.min_score > 0.0:
                all_results = [r for r in all_results if r.get("score", 0.0) >= payload.min_score]

            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            result = {
                "results": all_results[:payload.k],
                "memories": all_results[:payload.k]
            }
            if failed_shards:
                result["partial"] = True
                result["failed_shards"] = failed_shards
            return result
        else:
            if ctx.namespace in ("*", "all"):
                target_namespaces = get_namespaces_for_tenant(ctx.tenant)
                if not target_namespaces:
                    return {"results": [], "memories": []}

                # Pre-embed query once for fast cosine similarity scoring
                q_emb = None
                try:
                    embed_engine = db or (await get_db_instance(None, None))
                    e_backend = await embed_engine._get_db()
                    embedder = e_backend._get_embedder()
                    raw_q = await run_embedding(
                        embedder.encode,
                        query_text,
                        is_query=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    q_emb = np.array(raw_q, dtype=np.float32)
                    if len(q_emb) > e_backend.dim:
                        q_emb = q_emb[:e_backend.dim]
                        norm = np.linalg.norm(q_emb)
                        if norm > 1e-10:
                            q_emb /= norm
                except Exception as e:
                    logger.warning(f"Error pre-embedding query text: {e}")

                all_results = []
                seen_ids = set()
                for ns in target_namespaces:
                    try:
                        active_db = await get_db_instance(ctx.tenant, ns)
                        query_kwargs = {
                            "text": query_text,
                            "k": payload.k,
                            "filters": payload.filters,
                            "memory_type": payload.memory_type,
                            "context_window": payload.context_window,
                            "min_score": payload.min_score,
                        }
                        if payload.expand_hops > 0:
                            query_kwargs["expand_hops"] = payload.expand_hops
                        if payload.cold_search_mode:
                            query_kwargs["cold_search_mode"] = payload.cold_search_mode

                        try:
                            results = await active_db.query(**query_kwargs)
                        except TypeError:
                            query_kwargs.pop("expand_hops", None)
                            query_kwargs.pop("cold_search_mode", None)
                            results = await active_db.query(**query_kwargs)

                        for r in results:
                            if r.id not in seen_ids:
                                seen_ids.add(r.id)
                                score = float(getattr(r, "score", 0.0))
                                if score == 0.0 and q_emb is not None and q_emb.any() and hasattr(r, "_atom") and getattr(r._atom, "embedding", None) is not None and len(r._atom.embedding) > 0:
                                    atom_emb = r._atom.embedding
                                    min_len = min(len(atom_emb), len(q_emb))
                                    score = float(np.dot(atom_emb[:min_len], q_emb[:min_len]) / (
                                        np.linalg.norm(atom_emb[:min_len]) * np.linalg.norm(q_emb[:min_len]) + 1e-10
                                    ))
                                item_data = {
                                    "id": r.id,
                                    "text": r.text,
                                    "metadata": r.metadata,
                                    "created_at": r.created_at,
                                    "score": score
                                }
                                if hasattr(r._atom, "memory_type") and r._atom.memory_type:
                                    item_data["memory_type"] = r._atom.memory_type.value if hasattr(r._atom.memory_type, "value") else str(r._atom.memory_type)
                                all_results.append(item_data)
                    except Exception as e:
                        logger.warning(f"Error querying namespace {ns}: {e}")

                if payload.min_score > 0.0:
                    all_results = [r for r in all_results if r.get("score", 0.0) >= payload.min_score]

                all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                top_results = all_results[:payload.k]
                return {"results": top_results, "memories": top_results}
            else:
                active_db = await get_db_instance(ctx.tenant, ctx.namespace)
                try:
                    query_kwargs = {
                        "text": query_text,
                        "k": payload.k,
                        "filters": payload.filters,
                        "memory_type": payload.memory_type,
                        "context_window": payload.context_window,
                        "min_score": payload.min_score,
                    }
                    if payload.expand_hops > 0:
                        query_kwargs["expand_hops"] = payload.expand_hops
                    if payload.cold_search_mode:
                        query_kwargs["cold_search_mode"] = payload.cold_search_mode

                    try:
                        results = await active_db.query(**query_kwargs)
                    except TypeError:
                        query_kwargs.pop("expand_hops", None)
                        query_kwargs.pop("cold_search_mode", None)
                        results = await active_db.query(**query_kwargs)
                    
                    formatted_results = []
                    needs_rescoring = any(float(getattr(r, "score", 0.0)) == 0.0 for r in results)
                    q_emb = None
                    if needs_rescoring:
                        engine = await active_db._get_db()
                        embedder = engine._get_embedder()
                        raw_q = await run_embedding(
                            embedder.encode,
                            query_text,
                            is_query=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        q_emb = np.array(raw_q, dtype=np.float32)
                        if len(q_emb) > engine.dim:
                            q_emb = q_emb[:engine.dim]
                            norm = np.linalg.norm(q_emb)
                            if norm > 1e-10:
                                q_emb /= norm

                    for r in results:
                        score = float(getattr(r, "score", 0.0))
                        if score == 0.0 and q_emb is not None and q_emb.any() and getattr(r._atom, "embedding", None) is not None and r._atom.embedding.any():
                            atom_emb = r._atom.embedding
                            if len(atom_emb) != len(q_emb):
                                min_len = min(len(atom_emb), len(q_emb))
                                score = float(np.dot(atom_emb[:min_len], q_emb[:min_len]) / (
                                    np.linalg.norm(atom_emb[:min_len]) * np.linalg.norm(q_emb[:min_len]) + 1e-10
                                ))
                            else:
                                score = float(np.dot(atom_emb, q_emb) / (
                                    np.linalg.norm(atom_emb) * np.linalg.norm(q_emb) + 1e-10
                                ))
                        item_data = {
                            "id": r.id,
                            "text": r.text,
                            "metadata": r.metadata,
                            "created_at": r.created_at,
                            "score": score
                        }
                        if hasattr(r._atom, "memory_type") and r._atom.memory_type:
                            item_data["memory_type"] = r._atom.memory_type.value if hasattr(r._atom.memory_type, "value") else str(r._atom.memory_type)
                        formatted_results.append(item_data)

                    if payload.min_score > 0.0:
                        formatted_results = [r for r in formatted_results if r.get("score", 0.0) >= payload.min_score]

                    return {"results": formatted_results, "memories": formatted_results}
                except Exception as e:
                    logger.error(f"Error resolving retrieval operations: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("query", payload, request, response, ctx, fetch)

@app.post("/multi_hop")
async def multi_hop(
    payload: MultiHopPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Executes a multi-hop relational retrieval query across knowledge-graph associations and semantic embeddings.
    Coordinator fans out to shards and aggregates candidate results.
    """
    query_text = payload.get_query_text()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text must not be empty.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            forward_data = payload.model_dump()
            forward_data["text"] = query_text
            forward_data["query"] = query_text

            tasks = [
                client.post(
                    f"{node}/multi_hop",
                    json=forward_data,
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            all_results = []
            seen_ids = set()
            failed_shards = 0
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    memories = data.get("memories") or data.get("results") or []
                    for m in memories:
                        mid = m.get("id")
                        if mid and mid not in seen_ids:
                            seen_ids.add(mid)
                            all_results.append(m)
                        elif not mid:
                            all_results.append(m)
                else:
                    failed_shards += 1
                    logger.error(f"Error querying multi_hop on shard {node}: {format_shard_error(resp)}")

            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            result = {
                "results": all_results[:payload.k],
                "memories": all_results[:payload.k]
            }
            if failed_shards:
                result["partial"] = True
                result["failed_shards"] = failed_shards
            return result
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                kwargs = {
                    "hops": payload.hops,
                    "k": payload.k,
                    "filters": payload.filters,
                    "context_window": payload.context_window,
                }
                if payload.cold_search_mode:
                    kwargs["cold_search_mode"] = payload.cold_search_mode

                results = await active_db.multi_hop(query_text, **kwargs)
                formatted = []
                for r in results:
                    atom_dict = r._atom.to_dict()
                    if hasattr(r, "score"):
                        atom_dict["score"] = float(r.score)
                    formatted.append(atom_dict)
                return {"results": formatted, "memories": formatted}
            except Exception as e:
                logger.error(f"Error resolving multi_hop retrieval: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("multi_hop", payload, request, response, ctx, fetch)

@app.post("/skills/remember", status_code=status.HTTP_201_CREATED)
async def remember_skill(
    payload: SkillCreatePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Registers a synthesized procedural skill with steps, tool schema, and decision rules.
    Stored under MemoryType.SKILL and indexed in the Hot Tier SkillIndex.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        route_key = payload.skill_id or payload.skill_name
        target_shard = get_healthy_shard(route_key)
        shard_idx = shard_nodes.index(target_shard)
        stable_id = payload.skill_id or f"shard{shard_idx}-skill-{uuid.uuid4().hex}"
        forward_data = payload.model_dump()
        forward_data["skill_id"] = stable_id

        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/skills/remember",
                json=forward_data,
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        failures = []
        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 201:
                success_nodes.append(node)
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {"status": "offline"}

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required:
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "skill_registered", "id": stable_id})
            return {"status": "success", "id": stable_id, "skill_name": payload.skill_name}
        else:
            last_err = failures[-1][1] if failures else "Unknown error"
            raise HTTPException(
                status_code=503,
                detail=f"Skill replication failed for group {target_shard}. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            norm_steps = []
            for i, st in enumerate(payload.steps or []):
                if isinstance(st, str):
                    norm_steps.append({"step": i + 1, "action": st})
                elif isinstance(st, dict):
                    norm_steps.append(st)
                else:
                    norm_steps.append({"step": i + 1, "action": str(st)})

            aid = await active_db.remember_skill(
                skill_name=payload.skill_name,
                description=payload.description,
                steps=norm_steps,
                tool_schema=payload.tool_schema,
                metadata=payload.metadata,
                skill_id=payload.skill_id,
                decision_rules=payload.decision_rules,
                triples=payload.triples,
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "skill_registered", "id": aid})
            return {"status": "success", "id": aid, "skill_name": payload.skill_name}
        except Exception as e:
            logger.error(f"Error registering skill {payload.skill_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def _execute_get_skill(
    name: str,
    request: Request,
    response: Response,
    ctx: RequestContext
):
    if not name:
        raise HTTPException(status_code=400, detail="Skill name is required.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_node = hash_ring.get_node(name)
            group = get_replica_group_for_node(target_node)
            probe_node = get_healthy_replica_for_group(group) or group[0]

            try:
                res = await client.get(f"{probe_node}/skills/{name}", headers=get_forward_headers(ctx))
                if res.status_code == 200:
                    data = res.json()
                    if data and data.get("id"):
                        return data
            except Exception:
                pass

            target_nodes = []
            for grp in shard_groups:
                nd = get_healthy_replica_for_group(grp)
                if nd and nd != probe_node:
                    target_nodes.append(nd)

            if target_nodes:
                tasks = [client.get(f"{node}/skills/{name}", headers=get_forward_headers(ctx)) for node in target_nodes]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for resp in responses:
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        if data and data.get("id"):
                            return data
            return {}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mem = await active_db.get_skill(name)
                if mem:
                    return mem._atom.to_dict()
                return {}
            except Exception as e:
                logger.error(f"Error fetching skill {name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_skill", {"skill_name": name}, request, response, ctx, fetch)

@app.get("/skills/{skill_name}")
async def get_skill_by_path(
    skill_name: str,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a procedural skill by name via path parameter.
    """
    return await _execute_get_skill(skill_name, request, response, ctx)

@app.post("/skills/get")
async def get_skill_by_payload(
    payload: SkillGetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves a procedural skill by name via JSON payload.
    """
    return await _execute_get_skill(payload.skill_name, request, response, ctx)

@app.get("/skills")
async def list_skills(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Lists all procedural skills registered across the cluster.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            tasks = [client.get(f"{node}/skills", headers=get_forward_headers(ctx)) for node in target_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            all_skills = []
            seen = set()
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    for s in data.get("skills", []):
                        sid = s.get("id") or s.get("metadata", {}).get("skill_name")
                        if sid not in seen:
                            seen.add(sid)
                            all_skills.append(s)
            return {"skills": all_skills}
        else:
            if ctx.namespace in ("*", "all"):
                target_namespaces = get_namespaces_for_tenant(ctx.tenant)
                all_skills = []
                seen = set()
                for ns in target_namespaces:
                    try:
                        adb = await get_db_instance(ctx.tenant, ns)
                        s_list = await adb.list_skills()
                        for s in s_list:
                            s_dict = s._atom.to_dict() if hasattr(s, "_atom") else s.to_dict()
                            sid = s_dict.get("id") or s_dict.get("metadata", {}).get("skill_name")
                            if sid not in seen:
                                seen.add(sid)
                                all_skills.append(s_dict)
                    except Exception as e:
                        logger.warning(f"Error listing skills for ns {ns}: {e}")
                return {"skills": all_skills}
            else:
                active_db = await get_db_instance(ctx.tenant, ctx.namespace)
                try:
                    skills = await active_db.list_skills()
                    return {"skills": [s._atom.to_dict() for s in skills]}
                except Exception as e:
                    logger.error(f"Error listing skills: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("list_skills", {}, request, response, ctx, fetch)

@app.post("/profiles/remember", status_code=status.HTTP_201_CREATED)
async def remember_profile(
    payload: ProfileCreatePayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Stores long-term user profile facts and preferences under MemoryType.PROFILE.
    Coordinator routes deterministically by user_id.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        target_shard = get_healthy_shard(payload.user_id)
        group = get_replica_group_for_node(target_shard)
        tasks = [
            client.post(
                f"{node}/profiles/remember",
                json=payload.model_dump(),
                headers=get_forward_headers(ctx)
            ) for node in group
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        success_nodes = []
        result_id = None
        failures = []
        for node, resp in zip(group, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 201:
                success_nodes.append(node)
                result_id = resp.json().get("id")
            else:
                err_msg = format_shard_error(resp)
                failures.append((node, err_msg))
                if should_mark_shard_offline(resp):
                    SHARD_METRICS_CACHE[node] = {"status": "offline"}

        consistency = get_consistency_level(request)
        n = len(group)
        required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)

        if len(success_nodes) >= required:
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "profile_updated", "user_id": payload.user_id})
            return {"status": "success", "id": result_id, "user_id": payload.user_id}
        else:
            last_err = failures[-1][1] if failures else "Unknown error"
            raise HTTPException(
                status_code=503,
                detail=f"Profile replication failed. Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}",
                headers={"Retry-After": "5"}
            )
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            aid = await active_db.remember_user_profile(
                user_id=payload.user_id,
                fact_text=payload.fact_text,
                metadata=payload.metadata,
            )
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "profile_updated", "user_id": payload.user_id})
            return {"status": "success", "id": aid, "user_id": payload.user_id}
        except Exception as e:
            logger.error(f"Error remembering user profile for {payload.user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def _execute_get_profile(
    uid: str,
    request: Request,
    response: Response,
    ctx: RequestContext
):
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required.")

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_node = hash_ring.get_node(uid)
            group = get_replica_group_for_node(target_node)
            probe_node = get_healthy_replica_for_group(group) or group[0]

            res = await client.get(f"{probe_node}/profiles/{uid}", headers=get_forward_headers(ctx))
            if res.status_code == 200:
                return res.json()
            return {"memories": []}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                mems = await active_db.get_user_profile(uid)
                if not mems:
                    # Query with min_score=-1.0 to ensure profile facts aren't dropped by negative cosine angle
                    mems = await active_db.query(
                        text=f"user profile {uid}",
                        k=50,
                        filters={"user_id": uid},
                        memory_type="profile",
                        min_score=-1.0
                    )
                if not mems:
                    # Fallback to direct hot tier lookup by metadata
                    raw_db = await active_db._get_db()
                    matched_atoms = [
                        a for a in raw_db.hot_tier.atoms.values()
                        if a.memory_type and a.memory_type.value == "profile" and a.metadata.get("user_id") == uid
                    ]
                    return {"memories": [a.to_dict() for a in matched_atoms]}

                return {"memories": [m._atom.to_dict() if hasattr(m, "_atom") else m.to_dict() for m in mems]}
            except Exception as e:
                logger.error(f"Error fetching profile for {uid}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_profile", {"user_id": uid}, request, response, ctx, fetch)

@app.get("/profiles/{user_id}")
async def get_profile_by_path(
    user_id: str,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves all stored profile facts and preferences for a specific user via path.
    """
    return await _execute_get_profile(user_id, request, response, ctx)

@app.post("/profiles/get")
async def get_profile_by_payload(
    payload: ProfileGetPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves all stored profile facts and preferences for a specific user via JSON payload.
    """
    return await _execute_get_profile(payload.user_id, request, response, ctx)

async def _execute_summary_snapshot(
    uid: Optional[str],
    request: Request,
    response: Response,
    ctx: RequestContext
):
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            params = {"user_id": uid} if uid else {}
            tasks = [client.get(f"{node}/summary/snapshot", params=params, headers=get_forward_headers(ctx)) for node in target_nodes]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            snapshots = []
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    snap_text = resp.json().get("snapshot", "").strip()
                    if snap_text and snap_text not in snapshots:
                        snapshots.append(snap_text)
            aggregated = "\n\n".join(snapshots)
            return {"snapshot": aggregated}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                snap = await active_db.get_hot_summary_snapshot(uid)
                return {"snapshot": snap}
            except Exception as e:
                logger.error(f"Error generating summary snapshot: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("summary_snapshot", {"user_id": uid}, request, response, ctx, fetch)

@app.get("/summary/snapshot")
async def get_summary_snapshot(
    request: Request,
    response: Response,
    user_id: Optional[str] = Query(None),
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Produces a compact textual snapshot of active hot memories and profile facts via GET.
    """
    return await _execute_summary_snapshot(user_id, request, response, ctx)

@app.post("/summary/snapshot")
async def post_summary_snapshot(
    request: Request,
    response: Response,
    payload: Optional[SnapshotPayload] = None,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Produces a compact textual snapshot of active hot memories and profile facts via POST.
    """
    uid = payload.user_id if payload else None
    return await _execute_summary_snapshot(uid, request, response, ctx)

@app.post("/adaptive_query")
async def adaptive_query_memories(
    payload: AdaptiveQueryPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Intelligently routes a query to the optimal engine(s) (semantic, relational, temporal, or quantitative) 
    using LLM-orchestrated routing (or local rule fallbacks if offline) and retrieves relevant memories.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)
            tasks = [
                client.post(
                    f"{node}/adaptive_query",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    all_results.extend(data.get("results", []))
                elif isinstance(resp, Exception):
                    logger.error(f"Error adaptive querying shard: {resp}")
                    
            all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return {"results": all_results[:payload.k]}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                results = await active_db.adaptive_query(
                    query=payload.query,
                    k=payload.k,
                    context_window=payload.context_window
                )
                
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "id": r.id,
                        "text": r.text,
                        "metadata": r.metadata,
                        "created_at": r.created_at,
                        "score": 1.0
                    })
                return {"results": formatted_results}
            except Exception as e:
                logger.error(f"Error resolving adaptive query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("adaptive_query", payload, request, response, ctx, fetch)

@app.post("/update")
async def update_memory(payload: UpdatePayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.WRITE}))):
    """
    Updates memory text or metadata.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/update",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    logger.warning(f"Replication update failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }
                    
            consistency = get_consistency_level(request)
            n = len(group)
            if consistency == ConsistencyLevel.ALL:
                required_success = n
            elif consistency == ConsistencyLevel.QUORUM:
                required_success = n // 2 + 1
            else:
                required_success = 1
                
            if len(success_nodes) >= required_success:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
                return {"status": "success"}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Update consistency check failed. Required: {consistency.value} ({required_success}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            # Broadcast to all replica groups
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/update",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                    logger.error(f"Error updating replica {node}: {resp}")
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
            return {"status": "success"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            meta = dict(payload.metadata) if payload.metadata is not None else {}
            meta["_updated_at"] = time.time()
            if payload.triples is not None:
                meta["triples"] = payload.triples
            await active_db.update(payload.memory_id, payload.text, meta)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "update", "id": payload.memory_id})
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error updating memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_memory(payload: DeletePayload, request: Request, ctx: RequestContext = Depends(get_context({Permission.DELETE}))):
    """
    Deletes memory (hard or soft).
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_shard = get_shard_for_id(payload.memory_id)
        if target_shard:
            group = get_replica_group_for_node(target_shard)
            tasks = [
                client.post(
                    f"{node}/delete",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    logger.warning(f"Replication delete failed on replica {node}: {err_msg}")
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }
                    
            consistency = get_consistency_level(request)
            n = len(group)
            if consistency == ConsistencyLevel.ALL:
                required_success = n
            elif consistency == ConsistencyLevel.QUORUM:
                required_success = n // 2 + 1
            else:
                required_success = 1
                
            if len(success_nodes) >= required_success:
                invalidate_cache(ctx.tenant, ctx.namespace)
                await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
                return {"status": "success"}
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                raise HTTPException(
                    status_code=500,
                    detail=f"Delete consistency check failed. Required: {consistency.value} ({required_success}), got: {len(success_nodes)}. Last error: {last_err}"
                )
        else:
            # Broadcast to all replica groups
            target_nodes = []
            for group in shard_groups:
                target_nodes.extend(get_all_healthy_replicas_for_group(group))
            tasks = [
                client.post(
                    f"{node}/delete",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for node, resp in zip(target_nodes, responses):
                if isinstance(resp, Exception) or (isinstance(resp, httpx.Response) and resp.status_code != 200):
                    logger.error(f"Error deleting replica {node}: {resp}")
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
            return {"status": "success"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            await active_db.delete(payload.memory_id, payload.hard)
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "delete", "id": payload.memory_id})
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Error deleting memory {payload.memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def _prune_subgraph_by_depth(
    entity_id: str,
    edges: List[Dict[str, Any]],
    max_depth: int,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Prune edges and nodes to only those reachable from entity_id within max_depth hops.
    Matches entity_id case-insensitively if needed, but preserves canonical casing in graph.
    """
    if not edges:
        return {entity_id}, []

    start_node = entity_id
    ent_lower = entity_id.lower().strip()

    adj: Dict[str, Set[str]] = {}
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s and t:
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
            if s.lower().strip() == ent_lower and start_node != s:
                start_node = s
            elif t.lower().strip() == ent_lower and start_node != t:
                start_node = t

    if start_node not in adj:
        return {start_node}, []

    distances: Dict[str, int] = {start_node: 0}
    queue: deque = deque([start_node])
    while queue:
        curr = queue.popleft()
        d = distances[curr]
        if d >= max_depth:
            continue
        for nbr in adj.get(curr, ()):
            if nbr not in distances:
                distances[nbr] = d + 1
                queue.append(nbr)

    pruned_nodes = set(distances.keys())
    pruned_edges = [
        e for e in edges
        if e.get("source") in pruned_nodes and e.get("target") in pruned_nodes
    ]
    return pruned_nodes, pruned_edges


@app.get("/entity_graph")
async def entity_graph(
    request: Request,
    response: Response,
    entity_id: Optional[str] = None,
    depth: int = 2,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves the local entity graph or aggregates the distributed graph.
    """
    payload = {"entity_id": entity_id, "depth": depth}

    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)

            graph_timeout = COORDINATOR_GRAPH_TIMEOUT
            failed_shards: List[str] = []

            if not entity_id:
                tasks = [
                    client.get(
                        f"{node}/entity_graph",
                        params={"depth": depth},
                        headers=get_forward_headers(ctx),
                        timeout=graph_timeout,
                    ) for node in target_nodes
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                merged_nodes = set()
                merged_edges = []
                seen_edges = set()
                seen_relational_pairs = set()
                relational_merged = []
                fallback_merged = []
                merged_typed: Dict[str, Any] = {}

                for node, resp in zip(target_nodes, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        for n in data.get("nodes", []):
                            merged_nodes.add(n)
                        for edge in data.get("edges", []):
                            s = edge.get("source")
                            t = edge.get("target")
                            p = edge.get("predicate")
                            if not s or not t:
                                continue
                            pair = (s, t) if s <= t else (t, s)
                            key = (s, t, p, edge.get("memory_id"))
                            if key in seen_edges:
                                continue
                            seen_edges.add(key)
                            if p not in ("co_occurs_with", "mentions", "related"):
                                seen_relational_pairs.add(pair)
                                relational_merged.append(edge)
                            else:
                                fallback_merged.append((pair, edge))
                        for tn in data.get("typed_nodes") or []:
                            if isinstance(tn, dict) and tn.get("id"):
                                nid = str(tn["id"])
                                if nid not in merged_typed or (
                                    merged_typed[nid].get("type") in (None, "entity")
                                    and tn.get("type") not in (None, "entity")
                                ):
                                    merged_typed[nid] = tn
                        for nid, types in (data.get("node_types") or {}).items():
                            nid_str = str(nid)
                            if nid_str not in merged_typed:
                                merged_typed[nid_str] = {
                                    "id": nid_str,
                                    "types": types,
                                    "type": (types or [None])[0],
                                }
                            elif (
                                merged_typed[nid_str].get("type") in (None, "entity")
                                and types
                                and types[0] not in (None, "entity")
                            ):
                                merged_typed[nid_str]["types"] = types
                                merged_typed[nid_str]["type"] = types[0]
                    else:
                        failed_shards.append(node)
                        logger.error(
                            "Error querying entity graph from shard %s: %s",
                            node,
                            format_shard_error(resp),
                        )

                # Prioritize relational edges over 'related' or 'co_occurs_with'
                if relational_merged:
                    merged_edges = list(relational_merged)
                    rel_node_set = {e["source"] for e in merged_edges if e.get("source")} | {
                        e["target"] for e in merged_edges if e.get("target")
                    }
                    for nid, tn in merged_typed.items():
                        if tn.get("type") not in (None, "entity"):
                            rel_node_set.add(nid)
                    merged_nodes = rel_node_set
                else:
                    merged_edges = [edge for _, edge in fallback_merged]

                result: Dict[str, Any] = {"nodes": list(merged_nodes), "edges": merged_edges}
                if merged_typed:
                    result["typed_nodes"] = [tn for tn in merged_typed.values() if str(tn.get("id")) in merged_nodes]
                    result["node_types"] = {
                        tid: (tn.get("types") or [])
                        for tid, tn in merged_typed.items()
                        if tid in merged_nodes
                    }
                if failed_shards:
                    result["partial"] = True
                    result["failed_shards"] = failed_shards
                return result
            else:
                # One request per shard at full depth — shards already BFS locally.
                # The old hop×frontier×shard fan-out (depth=1 each) took tens of
                # seconds under embed load and timed out Percepta / Studio.
                hop_depth = max(1, min(depth, 10))
                tasks = [
                    client.get(
                        f"{node}/entity_graph",
                        params={"entity_id": entity_id, "depth": hop_depth},
                        headers=get_forward_headers(ctx),
                        timeout=graph_timeout,
                    )
                    for node in target_nodes
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                all_nodes = {entity_id}
                all_edges = []
                seen_edges = set()
                merged_typed: Dict[str, Any] = {}
                for node, resp in zip(target_nodes, responses):
                    if isinstance(resp, httpx.Response) and resp.status_code == 200:
                        data = resp.json()
                        for n in data.get("nodes", []):
                            all_nodes.add(n)
                        for edge in data.get("edges", []):
                            key = (
                                edge.get("source"),
                                edge.get("target"),
                                edge.get("predicate"),
                                edge.get("memory_id"),
                            )
                            if key not in seen_edges:
                                seen_edges.add(key)
                                all_edges.append(edge)
                        for tn in data.get("typed_nodes") or []:
                            if isinstance(tn, dict) and tn.get("id"):
                                nid = str(tn["id"])
                                if nid not in merged_typed or (
                                    merged_typed[nid].get("type") in (None, "entity")
                                    and tn.get("type") not in (None, "entity")
                                ):
                                    merged_typed[nid] = tn
                        for nid, types in (data.get("node_types") or {}).items():
                            nid_str = str(nid)
                            if nid_str not in merged_typed:
                                merged_typed[nid_str] = {
                                    "id": nid_str,
                                    "types": types,
                                    "type": (types or [None])[0],
                                }
                            elif (
                                merged_typed[nid_str].get("type") in (None, "entity")
                                and types
                                and types[0] not in (None, "entity")
                            ):
                                merged_typed[nid_str]["types"] = types
                                merged_typed[nid_str]["type"] = types[0]
                    else:
                        failed_shards.append(node)
                        logger.error(
                            "Error querying entity graph from shard %s: %s",
                            node,
                            format_shard_error(resp),
                        )

                candidate_edges = [
                    e for e in all_edges
                    if e.get("predicate") not in ("co_occurs_with", "mentions", "related")
                ]
                if not candidate_edges:
                    candidate_edges = all_edges

                final_nodes, final_edges = _prune_subgraph_by_depth(entity_id, candidate_edges, hop_depth)

                result = {"nodes": list(final_nodes), "edges": final_edges}
                if merged_typed:
                    result["typed_nodes"] = [tn for tn in merged_typed.values() if str(tn.get("id")) in final_nodes]
                    result["node_types"] = {
                        tid: (tn.get("types") or [])
                        for tid, tn in merged_typed.items()
                        if tid in final_nodes
                    }
                if failed_shards:
                    result["partial"] = True
                    result["failed_shards"] = failed_shards
                return result
        else:
            try:
                if not entity_id:
                    # Studio overview: cheap KG index walk. The old hub×BFS path
                    # called recall_by_entity per hub and wedged workers=1 shards
                    # under MiniLM load, so /entity_graph timed out → empty UI.
                    if ctx.namespace in ("*", "all"):
                        target_namespaces = [ns for ns in get_namespaces_for_tenant(ctx.tenant) if ns]
                        if not target_namespaces:
                            target_namespaces = ["default"]
                    else:
                        target_namespaces = [ctx.namespace]

                    if not target_namespaces:
                        return {"nodes": [], "edges": [], "typed_nodes": [], "node_types": {}}

                    engines = []
                    for ns in target_namespaces:
                        try:
                            adb = await get_db_instance(ctx.tenant, ns)
                            eng = await adb._get_db()
                            engines.append(eng)
                        except Exception as exc:
                            logger.warning(f"Failed to load engine for ns {ns}: {exc}")

                    if not engines:
                        return {"nodes": [], "edges": [], "typed_nodes": [], "node_types": {}}

                    def _overview_graph():
                        nodes = set()
                        edges = []
                        relational_edges = []
                        cooccur_edges = []
                        seen_relational = set()
                        seen_all_edges = set()
                        explicit_types: Dict[str, str] = {}
                        node_type_votes: Dict[str, Dict[str, int]] = {}

                        def _cast_vote(ent: str, vtype: Optional[str]):
                            if not ent or not vtype:
                                return
                            vtype_norm = normalize_type(vtype, allowed=ENTITY_TYPE_SET, default="")
                            if not vtype_norm or vtype_norm == "entity":
                                return
                            votes = node_type_votes.setdefault(ent, {})
                            votes[vtype_norm] = votes.get(vtype_norm, 0) + 1

                        for eng in engines:
                            with eng._internal_lock:
                                eng_atoms = list(eng.hot_tier.atoms.values())
                                try:
                                    for epoch_id in eng.cold_tier.get_all_epochs():
                                        eng_atoms.extend(eng.cold_tier.load_epoch(epoch_id))
                                except Exception as exc:
                                    logger.debug("Cold tier epoch load error: %s", exc)

                            for atom in eng_atoms:
                                if atom.id in getattr(eng, "deleted_atom_ids", set()):
                                    continue
                                if atom.metadata and atom.metadata.get("_deleted"):
                                    continue

                                mtype = str(getattr(atom.memory_type, "value", atom.memory_type) or "").lower()
                                if mtype == "skill" or (atom.metadata and atom.metadata.get("skill_name")):
                                    sk_name = (atom.metadata.get("skill_name") if atom.metadata else None) or atom.id
                                    explicit_types[sk_name] = "skill"
                                    nodes.add(sk_name)
                                elif mtype == "profile" or (atom.metadata and atom.metadata.get("user_id")):
                                    uid = atom.metadata.get("user_id") if atom.metadata else None
                                    if uid:
                                        explicit_types[uid] = "person"
                                        nodes.add(uid)
                                        role = atom.metadata.get("role")
                                        if role:
                                            explicit_types[role] = "role"
                                            nodes.add(role)
                                            relational_edges.append({
                                                "source": uid,
                                                "target": role,
                                                "predicate": "holds_role",
                                                "memory_id": atom.id,
                                            })
                                        team = atom.metadata.get("team")
                                        if team:
                                            explicit_types[team] = "organization"
                                            nodes.add(team)
                                            relational_edges.append({
                                                "source": uid,
                                                "target": team,
                                                "predicate": "member_of",
                                                "memory_id": atom.id,
                                            })

                                raw_triples = list(getattr(atom, "triples", []) or [])
                                if not raw_triples and atom.metadata:
                                    raw_triples = list(atom.metadata.get("triples") or [])

                                for item in raw_triples:
                                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                                        continue
                                    s, p, o = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
                                    if not s or not o:
                                        continue
                                    pred_slug = _slug(p)
                                    if not pred_slug:
                                        continue

                                    if pred_slug in TYPE_DEFINING_PREDICATES:
                                        canon = normalize_type(o, allowed=ENTITY_TYPE_SET, default="")
                                        if canon and canon != "entity":
                                            explicit_types[s] = canon
                                        nodes.add(s)
                                        nodes.add(o)
                                        relational_edges.append({
                                            "source": s,
                                            "target": o,
                                            "predicate": p,
                                            "memory_id": atom.id,
                                        })
                                        continue

                                    if pred_slug == "is_skill":
                                        explicit_types[s] = "skill"
                                        explicit_types[o] = "skill"
                                        nodes.add(s)
                                        nodes.add(o)
                                        relational_edges.append({
                                            "source": s,
                                            "target": o,
                                            "predicate": p,
                                            "memory_id": atom.id,
                                        })
                                        continue

                                    if pred_slug.startswith("has_step"):
                                        explicit_types[s] = "skill"
                                        _cast_vote(o, "process")
                                        nodes.add(s)
                                        nodes.add(o)
                                        relational_edges.append({
                                            "source": s,
                                            "target": o,
                                            "predicate": p,
                                            "memory_id": atom.id,
                                        })
                                        continue

                                    if pred_slug in PRED_ROLES:
                                        s_role, o_role = PRED_ROLES[pred_slug]
                                        _cast_vote(s, s_role)
                                        _cast_vote(o, o_role)

                                    if pred_slug not in ("co_occurs_with", "mentions", "related"):
                                        nodes.add(s)
                                        nodes.add(o)
                                        relational_edges.append({
                                            "source": s,
                                            "target": o,
                                            "predicate": p,
                                            "memory_id": atom.id,
                                        })
                                    else:
                                        cooccur_edges.append({
                                            "source": s,
                                            "target": o,
                                            "predicate": p,
                                            "memory_id": atom.id,
                                        })

                        if relational_edges:
                            for e in relational_edges:
                                s = e["source"]
                                t = e["target"]
                                key = (s, t, e.get("predicate"))
                                if key not in seen_all_edges:
                                    seen_all_edges.add(key)
                                    edges.append(e)
                            rel_nodes = {e["source"] for e in edges if e.get("source")} | {
                                e["target"] for e in edges if e.get("target")
                            }
                            for name, etype in explicit_types.items():
                                if etype in ("skill", "person", "organization", "process"):
                                    rel_nodes.add(name)
                            node_list = list(rel_nodes)
                        else:
                            for eng in engines:
                                hubs = eng.get_hub_entities(GRAPH_HUB_LIMIT)
                                if not hubs and not nodes:
                                    all_ents = eng.get_entities()
                                    if all_ents:
                                        hubs = all_ents[:GRAPH_HUB_LIMIT]

                                for ent in (hubs or []):
                                    if len(nodes) < GRAPH_OVERVIEW_NODE_LIMIT:
                                        nodes.add(ent)
                                    try:
                                        neighbors = eng.kg_manager.get_neighbors(ent)
                                    except Exception as exc:
                                        logger.warning("get_neighbors(%s) failed: %s", ent, exc)
                                        continue
                                    for n in neighbors:
                                        if len(nodes) < GRAPH_OVERVIEW_NODE_LIMIT:
                                            nodes.add(n)
                                        elif n not in nodes:
                                            continue
                                        pair = (ent, n) if ent <= n else (n, ent)
                                        key = (pair[0], pair[1], "related")
                                        if key not in seen_all_edges:
                                            seen_all_edges.add(key)
                                            edges.append({"source": ent, "target": n, "predicate": "related"})

                            for e in cooccur_edges:
                                s = e["source"]
                                t = e["target"]
                                if s in nodes and t in nodes:
                                    key = (s, t, e.get("predicate"))
                                    if key not in seen_all_edges:
                                        seen_all_edges.add(key)
                                        edges.append(e)
                            node_list = list(nodes)

                        typed_nodes = []
                        node_types: Dict[str, List[str]] = {}

                        for nid in node_list:
                            assigned_type = None
                            if nid in explicit_types:
                                assigned_type = explicit_types[nid]
                            elif nid in node_type_votes:
                                sorted_votes = sorted(
                                    node_type_votes[nid].items(),
                                    key=lambda kv: kv[1],
                                    reverse=True,
                                )
                                if sorted_votes:
                                    assigned_type = sorted_votes[0][0]

                            if not assigned_type:
                                inferred = infer_type_from_name(nid)
                                if inferred and inferred != "entity":
                                    assigned_type = inferred

                            if not assigned_type:
                                norm = normalize_type(nid, allowed=ENTITY_TYPE_SET, default="")
                                if norm and norm != "entity":
                                    assigned_type = norm

                            final_type = assigned_type or "entity"
                            node_types[nid] = [final_type]
                            typed_nodes.append({"id": nid, "types": [final_type], "type": final_type})

                        return {
                            "nodes": node_list,
                            "edges": edges,
                            "typed_nodes": typed_nodes,
                            "node_types": node_types,
                        }

                    return await asyncio.to_thread(_overview_graph)
                else:
                    hop_depth = max(1, min(depth, 10))
                    if ctx.namespace in ("*", "all"):
                        target_namespaces = get_namespaces_for_tenant(ctx.tenant)
                        all_candidate_edges = []
                        seen_candidate_keys = set()
                        raw_types: Dict[str, Any] = {}
                        registered: Set[str] = set()
                        for ns in target_namespaces:
                            try:
                                adb = await get_db_instance(ctx.tenant, ns)
                                graph = await adb.entity_graph(entity_id, hop_depth)
                                for edge in (graph.edges or []):
                                    ekey = (edge.get("source"), edge.get("target"), edge.get("predicate"))
                                    if ekey not in seen_candidate_keys:
                                        seen_candidate_keys.add(ekey)
                                        all_candidate_edges.append(edge)
                                if getattr(graph, "node_types", None):
                                    raw_types.update(graph.node_types)
                                registered.update(get_registered_ids(ctx.tenant, ns))
                            except Exception as exc:
                                logger.warning(f"Error querying entity graph for ns {ns}: {exc}")
                        candidate_edges = [
                            e for e in all_candidate_edges
                            if e.get("predicate") not in ("co_occurs_with", "mentions", "related")
                        ]
                        if not candidate_edges:
                            candidate_edges = all_candidate_edges
                        final_nodes, final_edges = _prune_subgraph_by_depth(entity_id, candidate_edges, hop_depth)
                    else:
                        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
                        graph = await active_db.entity_graph(entity_id, hop_depth)
                        all_edges = graph.edges or []
                        candidate_edges = [
                            e for e in all_edges
                            if e.get("predicate") not in ("co_occurs_with", "mentions", "related")
                        ]
                        if not candidate_edges:
                            candidate_edges = all_edges

                        final_nodes, final_edges = _prune_subgraph_by_depth(entity_id, candidate_edges, hop_depth)
                        raw_types = getattr(graph, "node_types", None) or {}
                        registered = set(get_registered_ids(ctx.tenant, ctx.namespace))

                    result = {"nodes": list(final_nodes), "edges": final_edges}
                    allowed = registered or None
                    raw_types = getattr(graph, "node_types", None) or {}
                    node_types: Dict[str, List[str]] = {}
                    typed_nodes = []

                    edge_votes: Dict[str, Dict[str, int]] = {}
                    explicit_types: Dict[str, str] = {}
                    for e in final_edges:
                        s = e.get("source")
                        t = e.get("target")
                        p = _slug(e.get("predicate"))
                        if p in TYPE_DEFINING_PREDICATES and t:
                            canon = normalize_type(t, allowed=allowed, default="")
                            if canon and canon != "entity":
                                explicit_types[s] = canon
                        elif p == "is_skill":
                            explicit_types[s] = "skill"
                            explicit_types[t] = "skill"
                        elif p.startswith("has_step"):
                            explicit_types[s] = "skill"
                            edge_votes.setdefault(t, {})["process"] = edge_votes.setdefault(t, {}).get("process", 0) + 1
                        elif p in PRED_ROLES:
                            s_role, o_role = PRED_ROLES[p]
                            if s_role and s:
                                edge_votes.setdefault(s, {})[s_role] = edge_votes.setdefault(s, {}).get(s_role, 0) + 1
                            if o_role and t:
                                edge_votes.setdefault(t, {})[o_role] = edge_votes.setdefault(t, {}).get(o_role, 0) + 1

                    for n in final_nodes:
                        assigned_type = None
                        if n in explicit_types:
                            assigned_type = explicit_types[n]
                        elif n in edge_votes:
                            sv = sorted(edge_votes[n].items(), key=lambda kv: kv[1], reverse=True)
                            if sv:
                                assigned_type = sv[0][0]
                        if not assigned_type and n in raw_types and raw_types[n]:
                            for rt in raw_types[n]:
                                norm = normalize_type(rt, allowed=allowed, default="")
                                if norm and norm != "entity":
                                    assigned_type = norm
                                    break
                        if not assigned_type:
                            inferred = infer_type_from_name(n, allowed=allowed)
                            if inferred and inferred != "entity":
                                assigned_type = inferred
                        if not assigned_type:
                            norm = normalize_type(n, allowed=ENTITY_TYPE_SET, default="")
                            if norm and norm != "entity":
                                assigned_type = norm

                        final_type = assigned_type or "entity"
                        node_types[n] = [final_type]
                        typed_nodes.append({"id": n, "types": [final_type], "type": final_type})

                    result["node_types"] = node_types
                    result["typed_nodes"] = typed_nodes
                    return result
            except Exception as e:
                logger.error(f"Error retrieving entity graph for {entity_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("entity_graph", payload, request, response, ctx, fetch)


def _merge_type_catalog(
    *,
    tenant: Optional[str],
    namespace: Optional[str],
    discovered: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge registered (app/seed) + auto-discovered types."""
    if namespace in ("*", "all"):
        registered = []
        seen_r = set()
        for ns in get_namespaces_for_tenant(tenant):
            for t in list_registered(tenant, ns):
                if t["id"] not in seen_r:
                    seen_r.add(t["id"])
                    registered.append(t)
    else:
        registered = list_registered(tenant, namespace)
    reg_ids = [t["id"] for t in registered]
    disc_types = []
    for t in discovered.get("types") or []:
        mapped = normalize_type(str(t), default="")
        if mapped and mapped != "entity" and mapped not in disc_types:
            disc_types.append(mapped)
        elif mapped == "entity":
            # Keep compact open ids that normalize_type accepted as-is via open path
            slug = str(t).strip().lower().replace(" ", "_").replace("-", "_")
            if slug and slug.replace("_", "").isalnum() and len(slug) <= 64 and slug not in disc_types:
                disc_types.append(slug)

    entity_types: Dict[str, List[str]] = {}
    for ent, tlist in (discovered.get("entity_types") or {}).items():
        cleaned = []
        for t in tlist or []:
            mapped = normalize_type(str(t), default="")
            if mapped and mapped != "entity" and mapped not in cleaned:
                cleaned.append(mapped)
        inferred = infer_type_from_name(str(ent))
        if inferred and inferred not in cleaned:
            cleaned.insert(0, inferred)
        if cleaned:
            entity_types[str(ent)] = cleaned

    # Union for clients that only read `types`
    union: List[str] = []
    for tid in reg_ids + disc_types:
        if tid not in union:
            union.append(tid)

    return {
        "types": union,
        "registered": registered,
        "registered_ids": reg_ids,
        "discovered": disc_types,
        "predicates": list(discovered.get("predicates") or []),
        "entity_types": entity_types,
        "discovery_source": discovered.get("source"),
        "closed": False,
        "seed_catalog": list(ENTITY_TYPES),
    }


async def _collect_discovered_types(ctx: RequestContext) -> Dict[str, Any]:
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        target_nodes = []
        for group in shard_groups:
            node = get_healthy_replica_for_group(group)
            if node:
                target_nodes.append(node)
        tasks = [
            client.get(
                f"{node}/discovered_types",
                headers=get_forward_headers(ctx),
                timeout=COORDINATOR_GRAPH_TIMEOUT,
            )
            for node in target_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        types: set = set()
        predicates: set = set()
        entity_types: Dict[str, set] = {}
        source = "shards"
        for resp in responses:
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                data = resp.json()
                # Prefer shard's discovered list when present (avoid re-merging registered)
                for t in data.get("discovered") or data.get("types") or []:
                    types.add(str(t))
                predicates.update(data.get("predicates") or [])
                for ent, tlist in (data.get("entity_types") or {}).items():
                    entity_types.setdefault(str(ent), set()).update(tlist or [])
                source = data.get("discovery_source") or source
        return {
            "types": sorted(types),
            "predicates": sorted(predicates),
            "entity_types": {k: sorted(v) for k, v in entity_types.items()},
            "source": source,
        }

    if ctx.namespace in ("*", "all"):
        target_namespaces = get_namespaces_for_tenant(ctx.tenant)
        all_types = set()
        all_predicates = set()
        all_entity_types = {}
        for ns in target_namespaces:
            try:
                adb = await get_db_instance(ctx.tenant, ns)
                eng = await adb._get_db()
                disc = await asyncio.to_thread(discover_types, eng)
                all_types.update(disc.get("types") or [])
                all_predicates.update(disc.get("predicates") or [])
                for ent, tlist in (disc.get("entity_types") or {}).items():
                    all_entity_types.setdefault(str(ent), set()).update(tlist or [])
            except Exception as exc:
                logger.warning(f"Error collecting discovered types for ns {ns}: {exc}")
        return {
            "types": sorted(all_types),
            "predicates": sorted(all_predicates),
            "entity_types": {k: sorted(v) for k, v in all_entity_types.items()},
            "source": "all_namespaces",
        }

    active_db = await get_db_instance(ctx.tenant, ctx.namespace)
    engine = await active_db._get_db()
    return await asyncio.to_thread(discover_types, engine)


@app.get("/discovered_types")
async def discovered_types(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Auto-discovered types from triples, merged with any app-registered types.
    Discovery stays open; apps predefine/create via /entity_types.
    """
    async def fetch():
        discovered = await _collect_discovered_types(ctx)
        return _merge_type_catalog(
            tenant=ctx.tenant, namespace=ctx.namespace, discovered=discovered
        )

    return await handle_cached_read("discovered_types", {}, request, response, ctx, fetch)


@app.get("/entity_types")
async def get_entity_types(
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ})),
):
    """List registered + discovered entity types for this tenant/namespace."""
    return await discovered_types(request, response, ctx)


class EntityTypePayload(BaseModel):
    id: str = Field(..., description="Type id (lowercase snake_case).")
    label: Optional[str] = Field(default=None, description="Human label.")
    description: Optional[str] = Field(default=None, description="Optional description.")


class EntityTypeSeedPayload(BaseModel):
    types: List[Any] = Field(
        ...,
        description="List of type ids or {id,label,description} objects.",
    )
    replace: bool = Field(
        default=False,
        description="If true, replace the registry with this list.",
    )
    use_default_seed: bool = Field(
        default=False,
        description="If true, seed the built-in suggested catalog (~22 types).",
    )


@app.post("/entity_types")
async def create_entity_type(
    payload: EntityTypePayload,
    ctx: RequestContext = Depends(get_context({Permission.WRITE})),
):
    """Register (create/update) an entity type for this tenant/namespace."""
    try:
        entry = registry_upsert_type(
            payload.id,
            label=payload.label,
            description=payload.description,
            source="app",
            tenant=ctx.tenant,
            namespace=ctx.namespace,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    invalidate_cache(ctx.tenant, ctx.namespace)
    return {"status": "success", "type": entry}


@app.put("/entity_types/seed")
async def seed_entity_types(
    payload: EntityTypeSeedPayload,
    ctx: RequestContext = Depends(get_context({Permission.WRITE})),
):
    """
    Bulk-predefine entity types. Pass use_default_seed=true for the suggested
    catalog, or supply types explicitly from the app.
    """
    entries = seed_entries() if payload.use_default_seed else payload.types
    if not entries:
        raise HTTPException(status_code=400, detail="No types to seed")
    try:
        result = registry_seed_types(
            entries,
            replace=payload.replace,
            source="seed",
            tenant=ctx.tenant,
            namespace=ctx.namespace,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    invalidate_cache(ctx.tenant, ctx.namespace)
    return result


@app.delete("/entity_types/{type_id}")
async def delete_entity_type(
    type_id: str,
    ctx: RequestContext = Depends(get_context({Permission.WRITE})),
):
    """Remove a registered type (does not erase discovered triples)."""
    ok = registry_delete_type(type_id, tenant=ctx.tenant, namespace=ctx.namespace)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Type '{type_id}' not registered")
    invalidate_cache(ctx.tenant, ctx.namespace)
    return {"status": "success", "deleted": type_id}

@app.post("/get_timeline")
async def get_timeline(
    payload: TimelinePayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Retrieves timeline chronologically.
    """
    async def fetch():
        if NODE_MODE == "coordinator":
            if not client:
                raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
                
            target_nodes = []
            for group in shard_groups:
                node = get_healthy_replica_for_group(group)
                if node:
                    target_nodes.append(node)
            tasks = [
                client.post(
                    f"{node}/get_timeline",
                    json=payload.model_dump(),
                    headers=get_forward_headers(ctx)
                ) for node in target_nodes
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_memories = []
            seen_ids = set()
            for resp in responses:
                if isinstance(resp, httpx.Response) and resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("memories", []):
                        if m.get("id") not in seen_ids:
                            seen_ids.add(m.get("id"))
                            all_memories.append(m)
                elif isinstance(resp, Exception):
                    logger.error(f"Error getting timeline from shard: {resp}")
                    
            all_memories.sort(key=lambda x: x.get("created_at", 0.0))
            return {"memories": all_memories}
        else:
            active_db = await get_db_instance(ctx.tenant, ctx.namespace)
            try:
                results = await active_db.get_timeline(entity_id=payload.entity_id, start=payload.start, end=payload.end)
                return {"memories": [r._atom.to_dict() for r in results]}
            except Exception as e:
                logger.error(f"Error getting timeline: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    return await handle_cached_read("get_timeline", payload, request, response, ctx, fetch)

@app.get("/stats")
async def stats(
    probe: Optional[int] = Query(
        None,
        description="If 1, return lightweight system metrics without opening the DB (for coordinator polls).",
    ),
    ctx: RequestContext = Depends(get_context({Permission.ADMIN})),
):
    """
    Provides real-time system metrics, cache status, and internal allocation maps.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")

        all_replica_nodes = []
        for group in shard_groups:
            all_replica_nodes.extend(group)

        # Prefer background poll cache when probe=True and every shard has an entry —
        # avoids live fan-out for health probes. Live client requests (probe=False) fetch fresh stats.
        if probe and SHARD_METRICS_CACHE and all(
            node in SHARD_METRICS_CACHE for node in all_replica_nodes
        ):
            total_memory_count = 0
            total_l1_size = 0
            total_l2_size = 0
            total_entity_count = 0
            accounted_groups = set()
            for idx, grp in enumerate(shard_groups):
                for node in grp:
                    metrics = SHARD_METRICS_CACHE.get(node)
                    if metrics and metrics.get("status") == "healthy" and idx not in accounted_groups:
                        accounted_groups.add(idx)
                        total_memory_count += metrics.get("memory_count", 0)
                        total_l1_size += metrics.get("l1_size", 0)
                        total_l2_size += metrics.get("l2_size", 0)
                        total_entity_count += metrics.get("entity_count", 0)
                        break
            shards_metrics = {node: dict(SHARD_METRICS_CACHE.get(node) or {}) for node in all_replica_nodes}
            return {
                "mode": "coordinator",
                "memory_count": total_memory_count,
                "l1_size": total_l1_size,
                "l2_size": total_l2_size,
                "entity_count": total_entity_count,
                "system": {
                    "cpu": get_cpu_usage(),
                    "ram": get_ram_usage(),
                    "disk": get_disk_usage(os.getenv("STORAGE_DIR", ".")),
                },
                "shards": shards_metrics,
                "cached": True,
            }

        tasks = [
            client.get(
                f"{node}/stats",
                params={"probe": "1"},
                headers=get_forward_headers(ctx),
                timeout=float(os.getenv("COORDINATOR_STATS_TIMEOUT", "15")),
            ) for node in all_replica_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_memory_count = 0
        total_l1_size = 0
        total_l2_size = 0
        total_entity_count = 0
        accounted_groups = set()
        shards_metrics = {}
        for node, resp in zip(all_replica_nodes, responses):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                try:
                    data = resp.json()
                    group_idx = None
                    for idx, grp in enumerate(shard_groups):
                        if node in grp:
                            group_idx = idx
                            break
                    if group_idx is not None and group_idx not in accounted_groups:
                        accounted_groups.add(group_idx)
                        total_memory_count += data.get("memory_count", 0)
                        total_l1_size += data.get("l1_size", 0)
                        total_l2_size += data.get("l2_size", 0)
                        total_entity_count += data.get("entity_count", 0)
                    
                    cached_status = SHARD_METRICS_CACHE.get(node, {}).get("status", "healthy")
                    metric_entry = {
                        "status": cached_status,
                        "memory_count": data.get("memory_count", 0),
                        "l1_size": data.get("l1_size", 0),
                        "l2_size": data.get("l2_size", 0),
                        "entity_count": data.get("entity_count", 0),
                        "cpu": data.get("cpu", 0.0),
                        "ram": data.get("ram", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                        "disk": data.get("disk", {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0}),
                    }
                    shards_metrics[node] = metric_entry
                    SHARD_METRICS_CACHE[node] = metric_entry
                except Exception as e:
                    shards_metrics[node] = {
                        "status": "unhealthy",
                        "error": f"Failed to parse response: {str(e)}",
                        "cpu": 0.0,
                        "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                        "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                    }
            else:
                err_msg = format_shard_error(resp)
                shards_metrics[node] = {
                    "status": "unhealthy",
                    "error": err_msg,
                    "cpu": 0.0,
                    "ram": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                    "disk": {"total": 0.0, "available": 0.0, "used": 0.0, "percent": 0.0},
                }
                
        coord_system = {
            "cpu": get_cpu_usage(),
            "ram": get_ram_usage(),
            "disk": get_disk_usage(os.getenv("STORAGE_DIR", "."))
        }
        
        return {
            "mode": "coordinator",
            "memory_count": total_memory_count,
            "l1_size": total_l1_size,
            "l2_size": total_l2_size,
            "entity_count": total_entity_count,
            "system": coord_system,
            "shards": shards_metrics
        }
    else:
        # Fast shard probe for coordinator poll_shards_loop & health checks — never opens new DBs or runs embeddings.
        if probe:
            db_counts = get_shard_db_stats(ctx.tenant, ctx.namespace)
            return {
                "mode": "shard",
                "probe": True,
                "version": SERVER_VERSION,
                "memory_count": db_counts.get("memory_count", 0),
                "l1_size": db_counts.get("l1_size", 0),
                "l2_size": db_counts.get("l2_size", 0),
                "entity_count": db_counts.get("entity_count", 0),
                "cpu": get_cpu_usage(),
                "ram": get_ram_usage(),
                "disk": get_disk_usage(os.getenv("STORAGE_DIR", "./shared_memory")),
            }

        if ctx.namespace in ("*", "all"):
            target_namespaces = get_namespaces_for_tenant(ctx.tenant)
            total_memory_count = 0
            total_l1_size = 0
            total_l2_size = 0
            total_entity_count = 0
            total_pending = 0
            for ns in target_namespaces:
                try:
                    adb = await get_db_instance(ctx.tenant, ns)
                    s = await adb.stats()
                    total_memory_count += s.get("memory_count", 0)
                    total_l1_size += s.get("l1_size", 0)
                    total_l2_size += s.get("l2_size", 0)
                    total_entity_count += s.get("entity_count", 0)
                    try:
                        total_pending += await adb.pending_extractions()
                    except Exception:
                        pass
                except Exception as exc:
                    logger.warning(f"Failed to get stats for ns {ns}: {exc}")
            db_stats = {
                "mode": "shard",
                "version": SERVER_VERSION,
                "memory_count": total_memory_count,
                "l1_size": total_l1_size,
                "l2_size": total_l2_size,
                "entity_count": total_entity_count,
                "pending_extractions": total_pending,
                "cpu": get_cpu_usage(),
                "ram": get_ram_usage(),
                "disk": get_disk_usage(os.getenv("STORAGE_DIR", "./shared_memory")),
            }
            return db_stats

        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            db_stats = await active_db.stats()
            db_stats["mode"] = "shard"
            db_stats["version"] = SERVER_VERSION
            try:
                db_stats["pending_extractions"] = await active_db.pending_extractions()
            except Exception:
                db_stats["pending_extractions"] = 0
            db_stats["cpu"] = get_cpu_usage()
            db_stats["ram"] = get_ram_usage()
            db_stats["disk"] = get_disk_usage(os.getenv("STORAGE_DIR", "./shared_memory"))
            for key in ("memory_count", "l1_size", "l2_size", "entity_count"):
                if key in db_stats:
                    _LAST_SHARD_DB_STATS[key] = db_stats[key]
            return db_stats
        except Exception as e:
            logger.error(f"Unable to safely pull analytical parameters: {str(e)}")
            raise HTTPException(status_code=500, detail="Stats access blocked.")


# -------------------------------------------------------------------------
# Granular Import / Export Endpoints
# -------------------------------------------------------------------------
@app.get("/export")
async def export_data_get(
    request: Request,
    response: Response,
    data_type: str = Query("all", description="Category to export: 'all', 'skills', 'profiles', 'triplets', or 'memories'."),
    memory_type: Optional[str] = Query(None, description="Optional memory type filter."),
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Granularly exports stored database content (skills, user profiles, knowledge graph triplets, or memories).
    """
    return await _execute_export(data_type, memory_type, ctx)

@app.post("/export")
async def export_data_post(
    payload: ExportPayload,
    request: Request,
    response: Response,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Granularly exports stored database content via POST payload.
    """
    return await _execute_export(payload.data_type, payload.memory_type, ctx)

async def _execute_export(
    data_type: str,
    memory_type: Optional[str],
    ctx: RequestContext
) -> Dict[str, Any]:
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        
        target_nodes = []
        for group in shard_groups:
            node = get_healthy_replica_for_group(group)
            if node:
                target_nodes.append(node)
                
        payload = {"data_type": data_type}
        if memory_type:
            payload["memory_type"] = memory_type
            
        tasks = [
            client.post(
                f"{node}/export",
                json=payload,
                headers=get_forward_headers(ctx)
            ) for node in target_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        merged_skills = []
        seen_skills = set()
        merged_profiles = []
        seen_profiles = set()
        merged_memories = []
        seen_memories = set()
        merged_triplets = []
        seen_triplets = set()
        
        for resp in responses:
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                body = resp.json().get("data", {})
                for s in body.get("skills", []):
                    sid = s.get("id") or (s.get("metadata", {}).get("skill_name") if s.get("metadata") else None)
                    if sid and sid not in seen_skills:
                        seen_skills.add(sid)
                        merged_skills.append(s)
                for p in body.get("profiles", []):
                    pid = p.get("id") or ((p.get("metadata", {}).get("user_id", "") if p.get("metadata") else "") + ":" + (p.get("payload") or ""))
                    if pid and pid not in seen_profiles:
                        seen_profiles.add(pid)
                        merged_profiles.append(p)
                for m in body.get("memories", []):
                    mid = m.get("id")
                    if mid and mid not in seen_memories:
                        seen_memories.add(mid)
                        merged_memories.append(m)
                for t in body.get("triplets", []):
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        tkey = (str(t[0]), str(t[1]), str(t[2]))
                        if tkey not in seen_triplets:
                            seen_triplets.add(tkey)
                            merged_triplets.append(list(tkey))
                            
        return {
            "version": "1.0",
            "server_version": SERVER_VERSION,
            "exported_at": time.time(),
            "data_type": data_type,
            "tenant": ctx.tenant or "global",
            "namespace": ctx.namespace or "default",
            "counts": {
                "skills": len(merged_skills),
                "profiles": len(merged_profiles),
                "memories": len(merged_memories),
                "triplets": len(merged_triplets)
            },
            "data": {
                "skills": merged_skills,
                "profiles": merged_profiles,
                "memories": merged_memories,
                "triplets": merged_triplets
            }
        }
    else:
        if ctx.namespace in ("*", "all"):
            target_namespaces = get_namespaces_for_tenant(ctx.tenant)
            result = {
                "skills": [],
                "profiles": [],
                "memories": [],
                "triplets": []
            }
            seen_skills = set()
            seen_profiles = set()
            seen_memories = set()
            seen_triples = set()

            for ns in target_namespaces:
                try:
                    adb = await get_db_instance(ctx.tenant, ns)
                    raw_db = await adb._get_db()

                    if data_type in ("all", "skills"):
                        s_list = await adb.list_skills()
                        for s in s_list:
                            s_dict = s._atom.to_dict() if hasattr(s, "_atom") else s.to_dict()
                            sid = s_dict.get("id") or (s_dict.get("metadata", {}).get("skill_name") if s_dict.get("metadata") else None)
                            if sid and sid not in seen_skills:
                                seen_skills.add(sid)
                                result["skills"].append(s_dict)

                    ns_atoms = {}
                    with raw_db._internal_lock:
                        for aid, atom in raw_db.hot_tier.atoms.items():
                            if aid not in raw_db.deleted_atom_ids:
                                ns_atoms[aid] = atom
                        for epoch_id in raw_db.cold_tier.get_all_epochs():
                            for atom in raw_db.cold_tier.load_epoch(epoch_id):
                                if atom.id not in raw_db.deleted_atom_ids and atom.id not in ns_atoms:
                                    ns_atoms[atom.id] = atom

                    for atom in ns_atoms.values():
                        mt = atom.memory_type.value if atom.memory_type else "general"
                        atom_dict = atom.to_dict()

                        if data_type in ("all", "profiles") and (mt == "profile" or (atom.metadata and atom.metadata.get("user_id"))):
                            pid = atom.id or ((atom.metadata.get("user_id", "") if atom.metadata else "") + ":" + (atom.payload or ""))
                            if pid and pid not in seen_profiles:
                                seen_profiles.add(pid)
                                result["profiles"].append(atom_dict)

                        if data_type in ("all", "memories"):
                            if not memory_type or mt == memory_type:
                                if atom.id and atom.id not in seen_memories:
                                    seen_memories.add(atom.id)
                                    result["memories"].append(atom_dict)

                        if data_type in ("all", "triplets", "graph"):
                            if atom.triples:
                                for t in atom.triples:
                                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                                        tkey = (str(t[0]), str(t[1]), str(t[2]))
                                        if tkey not in seen_triples:
                                            seen_triples.add(tkey)
                                            result["triplets"].append(list(tkey))
                except Exception as exc:
                    logger.warning(f"Export error for ns {ns}: {exc}")

            return {
                "version": "1.0",
                "server_version": SERVER_VERSION,
                "exported_at": time.time(),
                "data_type": data_type,
                "tenant": ctx.tenant or "global",
                "namespace": ctx.namespace or "all",
                "counts": {
                    "skills": len(result["skills"]),
                    "profiles": len(result["profiles"]),
                    "memories": len(result["memories"]),
                    "triplets": len(result["triplets"])
                },
                "data": result
            }

        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            raw_db = await active_db._get_db()
            result = {
                "skills": [],
                "profiles": [],
                "memories": [],
                "triplets": []
            }
            
            # Skills
            if data_type in ("all", "skills"):
                skills = await active_db.list_skills()
                result["skills"] = [s._atom.to_dict() if hasattr(s, "_atom") else s.to_dict() for s in skills]

            # Atoms from hot tier and cold tier
            all_atoms = {}
            with raw_db._internal_lock:
                for aid, atom in raw_db.hot_tier.atoms.items():
                    if aid not in raw_db.deleted_atom_ids:
                        all_atoms[aid] = atom
                for epoch_id in raw_db.cold_tier.get_all_epochs():
                    for atom in raw_db.cold_tier.load_epoch(epoch_id):
                        if atom.id not in raw_db.deleted_atom_ids and atom.id not in all_atoms:
                            all_atoms[atom.id] = atom

            seen_triples = set()
            for atom in all_atoms.values():
                mt = atom.memory_type.value if atom.memory_type else "general"
                atom_dict = atom.to_dict()

                if data_type in ("all", "profiles") and (mt == "profile" or (atom.metadata and atom.metadata.get("user_id"))):
                    result["profiles"].append(atom_dict)

                if data_type in ("all", "memories"):
                    if not memory_type or mt == memory_type:
                        result["memories"].append(atom_dict)

                if data_type in ("all", "triplets", "graph"):
                    if atom.triples:
                        for t in atom.triples:
                            if isinstance(t, (list, tuple)) and len(t) >= 3:
                                tkey = (str(t[0]), str(t[1]), str(t[2]))
                                if tkey not in seen_triples:
                                    seen_triples.add(tkey)
                                    result["triplets"].append(list(tkey))

            return {
                "version": "1.0",
                "server_version": SERVER_VERSION,
                "exported_at": time.time(),
                "data_type": data_type,
                "tenant": ctx.tenant or "global",
                "namespace": ctx.namespace or "default",
                "counts": {
                    "skills": len(result["skills"]),
                    "profiles": len(result["profiles"]),
                    "memories": len(result["memories"]),
                    "triplets": len(result["triplets"])
                },
                "data": result
            }
        except Exception as e:
            logger.error(f"Export execution failure: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/import", status_code=status.HTTP_201_CREATED)
async def import_data(
    payload: ImportPayload,
    request: Request,
    ctx: RequestContext = Depends(get_context({Permission.WRITE}))
):
    """
    Granularly imports database records (skills, user profiles, memories, and triplets).
    """
    data = payload.data.get("data", payload.data)
    filter_type = payload.data_type

    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
        if not shard_nodes:
            raise HTTPException(status_code=500, detail="No shard nodes available.")

        counts = {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0}
        failed = {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0}
        errors: List[str] = []

        forward_headers = get_forward_headers(ctx)
        consistency = get_consistency_level(request)

        async def forward_to_replicas(target_node: str, endpoint: str, item_payload: dict, item_desc: str) -> bool:
            group = get_replica_group_for_node(target_node)
            if not group:
                errors.append(f"{item_desc}: No replicas found for node {target_node}")
                return False

            tasks = [
                client.post(
                    f"{node}{endpoint}",
                    json=item_payload,
                    headers=forward_headers
                ) for node in group
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            success_nodes = []
            failures = []
            for node, resp in zip(group, responses):
                if isinstance(resp, httpx.Response) and 200 <= resp.status_code < 300:
                    success_nodes.append(node)
                else:
                    err_msg = format_shard_error(resp)
                    failures.append((node, err_msg))
                    if should_mark_shard_offline(resp):
                        SHARD_METRICS_CACHE[node] = {
                            "status": "offline",
                            "cpu": 0.0,
                            "ram": {"percent": 0.0},
                            "disk": {"percent": 0.0}
                        }

            n = len(group)
            required = n if consistency == ConsistencyLevel.ALL else (n // 2 + 1 if consistency == ConsistencyLevel.QUORUM else 1)
            if len(success_nodes) >= required:
                return True
            else:
                last_err = failures[-1][1] if failures else "Unknown error"
                logger.warning(
                    f"Import replication failed for {item_desc}. "
                    f"Required: {consistency.value} ({required}), got: {len(success_nodes)}. Last error: {last_err}"
                )
                errors.append(f"{item_desc}: {last_err}")
                return False

        # 1. Skills
        if filter_type in (None, "all", "skills") and "skills" in data:
            for s in data["skills"]:
                meta = s.get("metadata") or {}
                skill_name = meta.get("skill_name") or s.get("skill_name") or s.get("id")
                desc = meta.get("description") or s.get("description") or s.get("payload") or ""
                steps = meta.get("steps") or s.get("steps") or []
                if skill_name:
                    sk_payload = {
                        "skill_name": skill_name,
                        "description": desc,
                        "steps": steps,
                        "tool_schema": meta.get("tool_schema"),
                        "metadata": meta,
                        "skill_id": s.get("id") or meta.get("process_id"),
                        "decision_rules": meta.get("decision_rules"),
                        "triples": s.get("triples") or meta.get("triples")
                    }
                    target_node = hash_ring.get_node(skill_name)
                    if target_node and await forward_to_replicas(target_node, "/skills/remember", sk_payload, f"skill '{skill_name}'"):
                        counts["skills"] += 1
                    else:
                        failed["skills"] += 1
                        if not target_node:
                            errors.append(f"skill '{skill_name}': No shard node available in hash ring")

        # 2. Profiles
        if filter_type in (None, "all", "profiles") and "profiles" in data:
            for p in data["profiles"]:
                meta = p.get("metadata") or {}
                uid = meta.get("user_id") or p.get("user_id")
                fact = p.get("payload") or p.get("text") or p.get("fact_text") or meta.get("fact_text") or ""
                if uid and fact:
                    pr_payload = {"user_id": uid, "fact_text": fact, "metadata": meta}
                    target_node = hash_ring.get_node(uid)
                    if target_node and await forward_to_replicas(target_node, "/profiles/remember", pr_payload, f"profile '{uid}'"):
                        counts["profiles"] += 1
                    else:
                        failed["profiles"] += 1
                        if not target_node:
                            errors.append(f"profile '{uid}': No shard node available in hash ring")

        # 3. Memories
        if filter_type in (None, "all", "memories") and "memories" in data:
            for m in data["memories"]:
                text = m.get("payload") or m.get("text")
                if text:
                    mem_payload = {
                        "text": text,
                        "metadata": m.get("metadata"),
                        "triples": m.get("triples"),
                        "memory_type": m.get("memory_type") or (m.get("metadata") or {}).get("memory_type"),
                        "id": m.get("id")
                    }
                    target_node = hash_ring.get_node(text)
                    if target_node and await forward_to_replicas(target_node, "/remember", mem_payload, f"memory '{text[:30]}'"):
                        counts["memories"] += 1
                    else:
                        failed["memories"] += 1
                        if not target_node:
                            errors.append(f"memory '{text[:30]}': No shard node available in hash ring")

        # 4. Triplets
        if filter_type in (None, "all", "triplets", "graph") and "triplets" in data:
            for t in data["triplets"]:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    text = f"{t[0]} {str(t[1]).replace('_', ' ')} {t[2]}."
                    tr_payload = {"text": text, "triples": [[t[0], t[1], t[2]]]}
                    target_node = hash_ring.get_node(text)
                    if target_node and await forward_to_replicas(target_node, "/remember", tr_payload, f"triplet '{t[0]}-{t[1]}-{t[2]}'"):
                        counts["triplets"] += 1
                    else:
                        failed["triplets"] += 1
                        if not target_node:
                            errors.append(f"triplet '{t[0]}-{t[1]}-{t[2]}': No shard node available in hash ring")

        total_imported = sum(counts.values())
        total_failed = sum(failed.values())

        if total_imported > 0:
            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "import", "counts": counts})

        status_str = "success" if total_failed == 0 else ("partial_success" if total_imported > 0 else "failed")
        resp_data = {
            "status": status_str,
            "imported": counts,
            "total_imported": total_imported,
            "failed": failed,
            "total_failed": total_failed,
        }
        if errors:
            resp_data["errors"] = errors
            resp_data["error"] = errors[0]
        return resp_data
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            counts = {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0}

            # 1. Skills
            if filter_type in (None, "all", "skills") and "skills" in data:
                for s in data["skills"]:
                    meta = s.get("metadata") or {}
                    skill_name = meta.get("skill_name") or s.get("skill_name") or s.get("id")
                    desc = meta.get("description") or s.get("description") or s.get("payload") or ""
                    steps = meta.get("steps") or s.get("steps") or []
                    norm_steps = []
                    for idx, st in enumerate(steps):
                        if isinstance(st, str):
                            norm_steps.append({"step": idx + 1, "action": st})
                        elif isinstance(st, dict):
                            norm_steps.append(st)
                        else:
                            norm_steps.append({"step": idx + 1, "action": str(st)})
                    if skill_name:
                        await active_db.remember_skill(
                            skill_name=skill_name,
                            description=desc,
                            steps=norm_steps,
                            tool_schema=meta.get("tool_schema"),
                            metadata=meta,
                            skill_id=s.get("id") or meta.get("process_id"),
                            decision_rules=meta.get("decision_rules"),
                            triples=s.get("triples") or meta.get("triples")
                        )
                        counts["skills"] += 1

            # 2. Profiles
            if filter_type in (None, "all", "profiles") and "profiles" in data:
                for p in data["profiles"]:
                    meta = p.get("metadata") or {}
                    uid = meta.get("user_id") or p.get("user_id")
                    fact = p.get("payload") or p.get("text") or p.get("fact_text") or meta.get("fact_text") or ""
                    if uid and fact:
                        await active_db.remember_user_profile(user_id=uid, fact_text=fact, metadata=meta)
                        counts["profiles"] += 1

            # 3. Memories
            if filter_type in (None, "all", "memories") and "memories" in data:
                for m in data["memories"]:
                    text = m.get("payload") or m.get("text")
                    if text:
                        await active_db.remember(
                            text=text,
                            triples=m.get("triples"),
                            metadata=m.get("metadata"),
                            memory_type=m.get("memory_type") or (m.get("metadata") or {}).get("memory_type"),
                            atom_id=m.get("id")
                        )
                        counts["memories"] += 1

            # 4. Triplets
            if filter_type in (None, "all", "triplets", "graph") and "triplets" in data:
                for t in data["triplets"]:
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        await active_db.remember(
                            text=f"{t[0]} {str(t[1]).replace('_', ' ')} {t[2]}.",
                            triples=[[t[0], t[1], t[2]]]
                        )
                        counts["triplets"] += 1

            invalidate_cache(ctx.tenant, ctx.namespace)
            await broadcast_sse_event("update", {"type": "import", "counts": counts})
            return {
                "status": "success",
                "imported": counts,
                "total_imported": sum(counts.values()),
                "failed": {"skills": 0, "profiles": 0, "memories": 0, "triplets": 0},
                "total_failed": 0
            }
        except Exception as e:
            logger.error(f"Import execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/compact", status_code=status.HTTP_200_OK)
async def compact(ctx: RequestContext = Depends(get_context({Permission.ADMIN}))):
    """
    Administrative endpoint to compress historical Parquet archives, clear soft deletes,
    and release unneeded disk space dynamically.
    """
    if NODE_MODE == "coordinator":
        if not client:
            raise HTTPException(status_code=503, detail="Coordinator HTTP client not ready.")
            
        target_nodes = []
        for group in shard_groups:
            target_nodes.extend(get_all_healthy_replicas_for_group(group))
        tasks = [
            client.post(
                f"{node}/compact",
                headers=get_forward_headers(ctx)
            ) for node in target_nodes
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for resp in responses:
            if isinstance(resp, Exception):
                logger.error(f"Error compacting shard: {resp}")
        invalidate_cache(ctx.tenant, ctx.namespace)
        return {"status": "compaction completed"}
    else:
        active_db = await get_db_instance(ctx.tenant, ctx.namespace)
        try:
            logger.info("Triggering background historical archive compaction...")
            await active_db.compact()
            invalidate_cache(ctx.tenant, ctx.namespace)
            return {"status": "compaction completed"}
        except Exception as e:
            logger.error(f"Compaction runtime error occurred: {str(e)}")
            raise HTTPException(status_code=500, detail="Compaction execution failure.")


class SQLQueryPayload(BaseModel):
    query: str


@app.post("/v1/analytics/query", status_code=status.HTTP_200_OK)
async def analytics_query(
    payload: SQLQueryPayload,
    ctx: RequestContext = Depends(get_context({Permission.READ}))
):
    """
    Executes a DuckDB SQL analytical query over Cold Tier Parquet archives.
    The table `cold_tier` is automatically available representing all parquet files.
    """
    active_db = await get_db_instance(ctx.tenant, ctx.namespace)
    try:
        results = await active_db.query_sql(payload.query)
        return {"status": "success", "data": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Analytics SQL query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL Query failed: {str(e)}")


UI_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "ui")
UI_ASSETS_DIR = os.path.join(UI_STATIC_DIR, "assets")

@app.get("/logo.png")
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "logo-epoch.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    absolute_logo = "/home/jeff/Projects/epochdb-server/src/logo-epoch.png"
    if os.path.exists(absolute_logo):
        return FileResponse(absolute_logo)
    raise HTTPException(status_code=404, detail="Logo not found")

@app.get("/favicon.svg")
@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = os.path.join(UI_STATIC_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

def _inject_studio_config(html_content: str) -> str:
    """Inject server mode/version/default tokens into the Studio SPA HTML."""
    inject_script = f"""<script>
        window.__EPOCHDB_SERVER_CONFIG__ = {{
            mode: "{NODE_MODE}",
            version: "{SERVER_VERSION}",
            defaultApiKey: "{API_KEY or ''}",
            defaultInternalToken: "{INTERNAL_AUTH_TOKEN or ''}"
        }};
        </script>"""
    if "<head>" in html_content:
        return html_content.replace("<head>", f"<head>\n    {inject_script}", 1)
    return inject_script + html_content

@app.get("/", response_class=HTMLResponse)
@app.get("/visualize", response_class=HTMLResponse)
async def visualize():
    index_path = os.path.join(UI_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(
            content=_inject_studio_config(html_content),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    raise HTTPException(
        status_code=404,
        detail="Studio UI not found. Build the React app with: cd ui && npm run build",
    )

@app.get("/stream")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint to stream database mutation notifications to clients.
    """
    queue = asyncio.Queue()
    SSE_LISTENERS.append(queue)
    
    async def event_generator():
        try:
            # Yield initial connection confirmation
            yield {"event": "connected", "data": "connected"}
            while True:
                event = await queue.get()
                sse_event = event.copy()
                if isinstance(sse_event.get("data"), (dict, list)):
                    sse_event["data"] = json.dumps(sse_event["data"])
                yield sse_event
        except asyncio.CancelledError:
            pass
        finally:
            if queue in SSE_LISTENERS:
                SSE_LISTENERS.remove(queue)
                
    return EventSourceResponse(event_generator())

# Mount Studio static assets after API routes so /assets does not shadow them.
if os.path.isdir(UI_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=UI_ASSETS_DIR), name="ui_assets")

# -------------------------------------------------------------------------
# 6. Production Execution Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
        workers=1,
        loop="auto",
        http="auto"
    )