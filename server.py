# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 5 MCP tools:
#     暴露 5 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import json
import hashlib
import hmac
import random
import logging
import asyncio
import contextvars
import threading
import base64
import mimetypes
import re
import time
import httpx
import jieba
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from uuid import uuid4
from datetime import datetime, timedelta, timezone

_BJ_TZ = timezone(timedelta(hours=8))
_WEEKDAYS_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _now_bj_header() -> str:
    """Beijing-time header for briefings — anchors the just-woken Claude
    in real time so it never falls back to LLM-imagined location/activity.
    Born from 5.9 + 5.10 'she's at her desk' hallucinations.
    """
    now = datetime.now(_BJ_TZ)
    return f"现在 {now.strftime('%Y-%m-%d')} {_WEEKDAYS_CN[now.weekday()]} {now.strftime('%H:%M')}"

# --- jieba 预热：避免首次 search 卡顿 / Pre-load jieba dict to avoid first-call lag ---
jieba.initialize()

# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from bucket_manager import BucketManager, bucket_revision_hash, literal_retrieval_keys
from dehydrator import (
    Dehydrator,
    SelfContainmentError,
    _safe_chat_completion_diagnostics,
)
from decay_engine import DecayEngine
from consolidation_engine import ConsolidationEngine
from episode_engine import EpisodeEngine
from saga_engine import SagaEngine
from sense_tagger import detect_senses, union_senses
from embedding_engine import EmbeddingEngine
from entity_store import EntityStore, entity_mention_present
from import_memory import ImportEngine
from intent_recall import bucket_intent_score_multiplier, resolve_intent_recall_policy
from fact_conflicts import build_supersedes_audit
from fact_slots import (
    FACT_STATUS_CURRENT,
    FACT_STATUS_HISTORICAL,
    STATE_VIEW_CURRENT,
    STATE_VIEW_HISTORICAL,
    STATE_VIEW_NEUTRAL,
    STATE_VIEW_TRANSITION,
    align_fact_state_candidates,
    fact_state_label,
    fact_slot_applies_to_bucket,
    filter_fact_slot_candidates,
    profile_fact_state_query,
    registered_fact_query_matches,
    registered_fact_key,
    state_link_target_ids,
)
from status_validity import (
    OperationalStatusValidityStore,
    STATE_CURRENT as OPERATIONAL_STATE_CURRENT,
    STATE_HISTORICAL as OPERATIONAL_STATE_HISTORICAL,
    VIEW_CURRENT as OPERATIONAL_VIEW_CURRENT,
    VIEW_NEUTRAL as OPERATIONAL_VIEW_NEUTRAL,
    is_operational_status_fact,
    operational_status_query_view,
    validity_label as operational_validity_label,
)
from query_expand import expand_query
from lmc5_recall_adapter import fuse_ranked_channels as lmc5_fuse_ranked_channels
from vendor.anchor_memory.recall_v2 import POLICIES as ANCHOR_RECALL_POLICIES
from recall_support import (
    expand_relation_graph,
    rank_within_relevance_bands,
    retain_original_query_supported_candidates,
)
from timeline_axis import timeline_neighbors
from recall_receipt import RecallReceiptConflict, RecallReceiptStore, normalize_bucket_ids
from e_axis_shadow import (
    EAxisShadowStore,
    build_failure_record,
    build_shadow_annotation,
    normalize_min_confidence,
    strict_json_loads,
)
from e_axis_recall import (
    apply_resonance_tie_break,
    derive_response_posture,
    format_response_posture,
    group_primary_authored_buckets,
    infer_query_emotion,
    load_e_axis_recall_config,
    rank_annotation_bucket_ids,
    resonance_score as e_axis_resonance_score,
    select_current_annotation,
)
from r2_storage import r2_storage
from sensory_engine import SensoryEngine, format_body_state_block, senses_from_sensory
from pg_mirror_queue import PgMirrorWorker
from utils import (
    load_config, setup_logging, strip_wikilinks, count_tokens_approx,
    world_matches, save_current_world, UNIVERSAL_WORLD, ResolvedGuardError,
    rrf_fuse, rrf_fuse_channels, parse_relative_time, PROTECTED_RESOLVE_DOMAINS,
    RELATION_TYPES, SAFE_RELATION_TYPES, REVIEW_RELATION_TYPES,
    PROPAGATION_RELATION_TYPES,
    event_at_from_metadata, now_iso,
)
from redact import redact_embedding_input, redact_text  # 只抹 secret，不审查情感内容
from mcp_auth import (
    APIBearerAuthMiddleware,
    MCPBearerAuthMiddleware,
    require_api_token,
    require_mcp_token,
)
from review_queue import (
    ReviewQueue, make_relation_entry, make_z_pair_entry,
    render_md as _render_review_md,
    KIND_CLOTHING, KIND_RELATION, KIND_Z_CONFLICT, KIND_METABOLISM, KIND_E_PROPOSAL,
    query_requests_history,
    rest_resolve_status_allowed,
)
from z_lifecycle import (
    ZLifecycleNotFound,
    ZLifecycleStateError,
    ZLifecycleTransaction,
)
from relation_approval import (
    RelationApprovalNotFound,
    RelationApprovalStateError,
    RelationApprovalTransaction,
)
from recall_history import (
    JsonFileRecallHistory,
    default_content_fingerprint,
    recall_identity,
)
from recall_timing import (
    begin_recall_timing,
    finish_recall_timing,
    get_recall_partial_result,
    finish_recall_stage,
    recall_stage,
    record_recall_dehydration,
    record_recall_metric,
    record_recall_stage,
    reset_recall_timing,
    set_recall_partial_result,
    start_recall_stage,
)
from lmc5_ledger import (
    LMC5Ledger,
    LedgerConflictError,
    LedgerCorruptionError,
    LedgerError,
    LedgerValidationError,
)
from lmc5_ingest_guard import (
    RawIngestBusy,
    RawIngestGuardError,
    shared_acceptance_write_guard,
    shared_ingest_guard,
)
from night_run_coordinator import NightRunCoordinatorError
from night_run_runtime import (
    NightRunRuntimeError,
    build_night_run_runtime,
)

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Initialize core components / 初始化核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
embedding_engine = EmbeddingEngine(config)            # Embedding engine / 向量化引擎
pg_mirror_worker = PgMirrorWorker(bucket_mgr.pg_mirror_queue)
consolidation_engine = ConsolidationEngine(config, bucket_mgr, embedding_engine)  # Consolidation engine / 整理引擎（夜班）
# Narrative layer (kernel 3): Event -> Episode -> Saga. Episode owns the loop,
# runs saga consolidation after building episodes each cycle.
# 叙事层（内核 3）：Event -> Episode -> Saga。episode 引擎持有后台循环，每轮卷完 episode 后接 saga 归并。
saga_engine = SagaEngine(config, bucket_mgr, dehydrator)
episode_engine = EpisodeEngine(config, bucket_mgr, embedding_engine, dehydrator, saga_engine=saga_engine)
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎
sensory_engine = SensoryEngine(config["buckets_dir"])  # External body-state sidecar / 外部身体状态层

# --- 待审队列（#2 Z轴事实演化 + #3 关系闸的共用 pending 存储）---
# 落在 <buckets_dir>/review_queue.jsonl。机器提议只进 pending；Z 冲突默认
# dry-run，只有显式 apply 才入队，只有显式人审事务才改事实 lifecycle。
_review_queue = None
_z_lifecycle_transaction = None
_relation_approval_transaction = None
_e_axis_shadow_store = None
_lmc5_ledger = None
_entity_store = None
_entity_store_key = None
_entity_store_initialized = False
_operational_status_validity_store = None
_entity_sync_locks: dict[tuple[str, str], object] = {}
_entity_sync_locks_guard = threading.Lock()
_recall_receipt_store = None
_recall_receipt_store_key = None
_lmc5_ledger_lock = threading.Lock()
_lmc5_night_runtime = None
_lmc5_night_runtime_lock = threading.Lock()
_strict_recall_errors = contextvars.ContextVar(
    "ombre_strict_recall_errors",
    default=False,
)
_breath_candidate_capture = contextvars.ContextVar(
    "ombre_breath_candidate_capture",
    default=None,
)


class RecallOperationalError(RuntimeError):
    """A required recall channel failed instead of returning valid evidence."""


def _get_recall_receipt_store() -> RecallReceiptStore:
    """Return the write-only final-injection ledger for this bucket vault."""
    global _recall_receipt_store, _recall_receipt_store_key
    key = os.path.abspath(str(config.get("buckets_dir") or ""))
    if _recall_receipt_store is None or _recall_receipt_store_key != key:
        store = RecallReceiptStore(key)
        store.initialize()
        _recall_receipt_store = store
        _recall_receipt_store_key = key
    return _recall_receipt_store


def _entity_registry_key() -> tuple[str, str]:
    """Stable key for the runtime entity sidecar and its audited seeds."""
    entities_cfg = config.get("entities", {}) or {}
    try:
        seed_key = json.dumps(
            entities_cfg.get("seeds") or [],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        seed_key = "[]"
    return os.path.abspath(str(config.get("buckets_dir") or "")), seed_key


def _get_entity_store(*, initialize: bool) -> EntityStore | None:
    """Return the additive entity sidecar without creating it on recall.

    ``breath`` always asks for ``initialize=False``.  A missing/corrupt sidecar
    therefore degrades to the legacy keyword/vector channels and never turns a
    read into a filesystem write.  Only hold/grow initialize schema and seeds.
    """
    global _entity_store, _entity_store_key, _entity_store_initialized
    entities_cfg = config.get("entities", {}) or {}
    if not entities_cfg.get("enabled", True):
        return None
    key = _entity_registry_key()
    if (
        _entity_store is None
        or _entity_store_key != key
        or (initialize and not _entity_store_initialized)
    ):
        try:
            _entity_store = EntityStore(config, initialize=initialize)
            _entity_store_key = key
            _entity_store_initialized = initialize
        except Exception as exc:
            logger.warning(
                "Entity sidecar unavailable; legacy channels remain active: %s",
                type(exc).__name__,
            )
            _entity_store = None
            _entity_store_key = key
            _entity_store_initialized = False
    return _entity_store


def _resolve_entity_recall(query: str) -> tuple[str, list[tuple[str, float]]]:
    """Resolve aliases and return a read-only third ranked channel."""
    store = _get_entity_store(initialize=False)
    if store is None:
        return query, []
    try:
        resolution = store.resolve_query(query)
        recall_query = getattr(resolution, "canonical_query", "") or query
        entity_ids = list(getattr(resolution, "entity_ids", ()) or ())
        linked_ids = store.linked_bucket_ids(entity_ids=entity_ids) if entity_ids else []
        return recall_query, [(bucket_id, 1.0) for bucket_id in linked_ids]
    except Exception as exc:
        logger.warning(
            "Entity recall failed open to legacy channels: %s",
            type(exc).__name__,
        )
        return query, []


def _entity_recall_settings() -> tuple[int, float]:
    """Parse optional channel tuning without letting bad config break recall."""
    entity_cfg = config.get("entities", {}) or {}
    try:
        top_k = int(entity_cfg.get("top_k", 20) or 20)
    except (TypeError, ValueError):
        top_k = 20
    try:
        weight = float(entity_cfg.get("rrf_weight", 1.0) or 0.0)
    except (TypeError, ValueError):
        weight = 1.0
    return max(1, min(top_k, 100)), max(0.0, weight)


def _link_bucket_entities(
    bucket_id: str,
    content: str,
    candidates: list[dict] | None = None,
) -> None:
    """Best-effort post-write link; bucket success never depends on sidecar."""
    entities_cfg = config.get("entities", {}) or {}
    clean_candidates = []
    for candidate in candidates or ():
        if not isinstance(candidate, dict):
            continue
        mention = candidate.get("mention")
        entity_type = candidate.get("type")
        if (
            isinstance(mention, str)
            and mention
            and entity_mention_present(content, mention)
            and entity_type in {"person", "place", "project"}
        ):
            clean_candidates.append({"mention": mention, "type": entity_type})

    should_initialize = bool(clean_candidates or (entities_cfg.get("seeds") or []))
    store = _get_entity_store(initialize=should_initialize)
    if store is None:
        return
    if not should_initialize and not os.path.isfile(store.db_path):
        # Generic deployments with no audited aliases and no prior entity DB
        # preserve the old zero-sidecar behavior.
        return
    try:
        store.resolve_and_link(bucket_id, content, candidates=clean_candidates)
    except ValueError as exc:
        # Model candidates are advisory.  If one still fails validation, retry
        # without them so audited seed scanning and the exact-content hash are
        # refreshed instead of leaving an older link permanently stale.
        if clean_candidates:
            logger.warning(
                "Entity candidates rejected for %s; retrying seed-only: %s",
                bucket_id,
                type(exc).__name__,
            )
            try:
                store.resolve_and_link(bucket_id, content, candidates=())
                return
            except Exception as retry_exc:
                exc = retry_exc
        logger.warning(
            "Entity link failed for %s; bucket write remains valid: %s",
            bucket_id,
            type(exc).__name__,
        )
    except Exception as exc:
        logger.warning(
            "Entity link failed for %s; bucket write remains valid: %s",
            bucket_id,
            type(exc).__name__,
        )


def _unlink_bucket_entities(bucket_id: str) -> None:
    """Best-effort sidecar cleanup after an authorized bucket deletion."""
    store = _get_entity_store(initialize=False)
    if store is None or not os.path.isfile(store.db_path):
        return
    try:
        store.unlink_bucket(bucket_id)
    except Exception as exc:
        logger.warning(
            "Entity unlink failed for %s: %s",
            bucket_id,
            type(exc).__name__,
        )


async def _synchronize_bucket_entities(
    bucket_id: str,
    content: str,
    candidates: list[dict] | None = None,
) -> None:
    """Link the latest bucket body, repairing post-write races safely.

    Bucket Markdown and the additive SQLite registry cannot share one atomic
    transaction.  Per-bucket serialization plus a read-before-link rule means
    a late older writer always links the current bucket body, never its stale
    argument.  Re-read after each sidecar commit to absorb bucket updates that
    land while this synchronizer is active.
    """
    requested_content = str(content or "")
    lock_key = (
        os.path.abspath(str(config.get("buckets_dir") or "")),
        str(bucket_id),
    )
    # MCP tools may run in worker threads with distinct event loops.  An
    # asyncio.Lock cached at module scope becomes bound to the first contended
    # loop and fails in the next one.  A process-wide threading.Lock is
    # loop-neutral; acquire it non-blockingly so a contending coroutine never
    # blocks its event-loop thread.
    with _entity_sync_locks_guard:
        lock = _entity_sync_locks.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _entity_sync_locks[lock_key] = lock
    while not lock.acquire(blocking=False):
        await asyncio.sleep(0.005)
    try:
        get_bucket = getattr(bucket_mgr, "get", None)
        if not callable(get_bucket):
            _link_bucket_entities(bucket_id, requested_content, candidates)
            return

        current_candidates = candidates
        for _attempt in range(3):
            try:
                latest = await get_bucket(bucket_id)
            except (AttributeError, NotImplementedError):
                _link_bucket_entities(bucket_id, requested_content, candidates)
                return
            except Exception as exc:
                logger.warning(
                    "Entity pre-link verification failed for %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
                return
            if not latest:
                _unlink_bucket_entities(bucket_id)
                return

            latest_content = str(latest.get("content") or "")
            if latest_content != requested_content:
                # Model spans were extracted from another revision.  The
                # audited seed catalog may still scan the actual latest body.
                current_candidates = None
            _link_bucket_entities(bucket_id, latest_content, current_candidates)

            try:
                verified = await get_bucket(bucket_id)
            except Exception as exc:
                logger.warning(
                    "Entity post-link verification failed for %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
                return
            if not verified:
                _unlink_bucket_entities(bucket_id)
                return
            verified_content = str(verified.get("content") or "")
            if verified_content == latest_content:
                return
            requested_content = verified_content
            current_candidates = None

        # A writer that changed the bucket during all three checks has its own
        # synchronizer queued on this same lock; it will read the latest body
        # before linking.  Until then the exact hash gate keeps recall safe.
        logger.warning("Entity link remained busy after bounded retries: %s", bucket_id)
    finally:
        lock.release()


# ImportEngine is constructed before helper definitions.  Attach the optional
# content callback here so raw/create/merge/recovery paths share the same
# entity synchronization boundary without changing any public API.
import_engine.content_sync = _synchronize_bucket_entities


def _get_review_queue() -> ReviewQueue:
    """懒构造，跟着运行时 buckets_dir 走（测试可改 config 后复位）。"""
    global _review_queue
    path = os.path.join(config["buckets_dir"], "review_queue.jsonl")
    if _review_queue is None or str(_review_queue.path) != os.path.abspath(path) \
            and str(_review_queue.path) != path:
        _review_queue = ReviewQueue(
            path,
            maintenance_root=config["buckets_dir"],
        )
    return _review_queue


def _get_z_lifecycle_transaction() -> ZLifecycleTransaction:
    """Lazily bind the Z approval transaction to the active test/prod vault."""
    global _z_lifecycle_transaction
    queue = _get_review_queue()
    root = os.path.abspath(config["buckets_dir"])
    if (
        _z_lifecycle_transaction is None
        or os.fspath(_z_lifecycle_transaction.root) != root
        or _z_lifecycle_transaction.bucket_manager is not bucket_mgr
        or _z_lifecycle_transaction.review_queue is not queue
    ):
        _z_lifecycle_transaction = ZLifecycleTransaction(
            root,
            bucket_mgr,
            queue,
        )
    return _z_lifecycle_transaction


def _get_relation_approval_transaction() -> RelationApprovalTransaction:
    """Lazily bind named relation approval to the active test/prod vault."""
    global _relation_approval_transaction
    queue = _get_review_queue()
    root = os.path.abspath(config["buckets_dir"])
    if (
        _relation_approval_transaction is None
        or os.fspath(_relation_approval_transaction.root) != root
        or _relation_approval_transaction.bucket_manager is not bucket_mgr
        or _relation_approval_transaction.review_queue is not queue
    ):
        _relation_approval_transaction = RelationApprovalTransaction(
            root,
            bucket_mgr,
            queue,
        )
    return _relation_approval_transaction


def _get_e_axis_shadow_store() -> EAxisShadowStore:
    global _e_axis_shadow_store
    path = os.path.join(config["buckets_dir"], ".axis", "e-shadow.jsonl")
    if _e_axis_shadow_store is None or str(_e_axis_shadow_store.path) != path:
        _e_axis_shadow_store = EAxisShadowStore(
            path,
            maintenance_root=config["buckets_dir"],
        )
    return _e_axis_shadow_store


def _get_lmc5_ledger() -> LMC5Ledger:
    """Return the append-only raw/chunk/candidate ledger for this vault."""
    global _lmc5_ledger
    path = os.path.join(
        config["buckets_dir"],
        ".lmc5",
        "pipeline.sqlite3",
    )
    expected_path = os.path.abspath(path)
    if _lmc5_ledger is None or os.fspath(_lmc5_ledger.path) != expected_path:
        with _lmc5_ledger_lock:
            if (
                _lmc5_ledger is None
                or os.fspath(_lmc5_ledger.path) != expected_path
            ):
                _lmc5_ledger = LMC5Ledger(
                    path,
                    maintenance_root=config["buckets_dir"],
                )
    return _lmc5_ledger


def _lmc5_night_enabled() -> bool:
    return os.environ.get("OMBRE_LMC5_NIGHT_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_lmc5_night_runtime():
    global _lmc5_night_runtime
    ledger = _get_lmc5_ledger()
    if (
        _lmc5_night_runtime is None
        or _lmc5_night_runtime.ledger is not ledger
    ):
        with _lmc5_night_runtime_lock:
            if (
                _lmc5_night_runtime is None
                or _lmc5_night_runtime.ledger is not ledger
            ):
                _lmc5_night_runtime = build_night_run_runtime(
                    config=config,
                    ledger=ledger,
                    bucket_manager=bucket_mgr,
                    embedding_engine=embedding_engine,
                    decay_engine=decay_engine,
                    consolidation_engine=consolidation_engine,
                )
    return _lmc5_night_runtime


async def _ensure_decay_background() -> None:
    if not _lmc5_night_enabled():
        await decay_engine.ensure_started()


async def _ensure_consolidation_background() -> None:
    if not _lmc5_night_enabled():
        await consolidation_engine.ensure_started()


def _hook_auth_state(request) -> str:
    """Authenticate private hook bridges without trusting proxy source IPs.

    The memory service is commonly reached through a local reverse proxy, so
    ``request.client.host`` cannot distinguish a real localhost hook from an
    Internet request.  A configured shared token is therefore mandatory.
    """
    expected = os.environ.get("OMBRE_HOOK_TOKEN", "")
    if not expected:
        return "unconfigured"
    supplied = ""
    try:
        supplied = str(request.headers.get("x-ombre-hook-token") or "")
    except Exception:
        supplied = ""
    if supplied and hmac.compare_digest(supplied, expected):
        return "authorized"
    return "forbidden"


async def _await_daemon_thread(function):
    """Await one blocking call without retaining a process-wide executor.

    The service is long-lived, while CLI/tests may import and exit immediately.
    A one-shot daemon worker avoids both event-loop blocking and a persistent
    default-executor thread keeping short-lived processes alive.
    """
    finished = threading.Event()
    outcome = {}

    def run():
        try:
            outcome["value"] = function()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True).start()
    # Polling avoids depending on a cross-thread selector wakeup after an
    # fsync-heavy SQLite call; the short sleep still yields the event loop.
    while not finished.is_set():
        await asyncio.sleep(0.01)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _fact_slot_registry() -> dict:
    cfg = config.get("fact_slots", {}) or {}
    if not cfg.get("enabled", False):
        return {}
    registry = cfg.get("registry", {}) or {}
    return registry if isinstance(registry, dict) else {}


def _operational_status_validity_enabled() -> bool:
    cfg = config.get("status_validity", {}) or {}
    return cfg.get("enabled", True) is True


def _get_operational_status_validity_store() -> OperationalStatusValidityStore:
    """Bind the additive validity sidecar to the active test/prod vault."""
    global _operational_status_validity_store
    path = os.path.abspath(os.path.join(
        config["buckets_dir"],
        ".validity",
        "operational_status.sqlite3",
    ))
    if (
        _operational_status_validity_store is None
        or _operational_status_validity_store.path != path
    ):
        _operational_status_validity_store = OperationalStatusValidityStore(path)
    return _operational_status_validity_store


def _state_recall_profile(query: str) -> dict:
    """Return the deterministic query-state profile and rollout switches."""
    profile = profile_fact_state_query(query, _fact_slot_registry())
    state_cfg = config.get("state_aware_recall", {}) or {}
    profile["enabled"] = state_cfg.get("enabled", True) is True
    profile["evidence_labels"] = state_cfg.get("evidence_labels", True) is True
    profile["operational_view"] = (
        operational_status_query_view(query)
        if _operational_status_validity_enabled()
        else OPERATIONAL_VIEW_NEUTRAL
    )
    try:
        link_limit = int(state_cfg.get("state_link_limit", 2))
    except (TypeError, ValueError):
        link_limit = 2
    profile["state_link_limit"] = min(3, max(0, link_limit))
    return profile


def _filter_z_fact_candidates(buckets, *, query: str, intent: str):
    """Apply the canonical Z gate only to exact-current fact questions."""
    candidates = list(buckets)
    profile = _state_recall_profile(query)
    if (
        _operational_status_validity_enabled()
        and profile["operational_view"] != OPERATIONAL_VIEW_NEUTRAL
    ):
        try:
            candidates = _get_operational_status_validity_store().attach(candidates)
        except Exception as exc:
            logger.warning(
                "Operational status validity lookup failed open: %s",
                type(exc).__name__,
            )
        if profile["operational_view"] == OPERATIONAL_VIEW_CURRENT:
            candidates = [
                bucket
                for bucket in candidates
                if operational_validity_label(
                    bucket,
                    view=profile["operational_view"],
                ).get("state") != OPERATIONAL_STATE_HISTORICAL
            ]
    if not profile["enabled"]:
        if query_requests_history(query):
            return candidates
        requested_keys = registered_fact_query_matches(
            query,
            _fact_slot_registry(),
        )
        return filter_fact_slot_candidates(
            candidates,
            intent=intent,
            registry=_fact_slot_registry(),
            fact_keys=requested_keys,
        )
    if (
        profile["view"] in {STATE_VIEW_HISTORICAL, STATE_VIEW_TRANSITION}
        or profile["historical_hints"]
    ):
        return candidates
    requested_keys = profile["fact_keys"] or registered_fact_query_matches(
        query,
        _fact_slot_registry(),
    )
    return filter_fact_slot_candidates(
        candidates,
        intent=intent,
        registry=_fact_slot_registry(),
        fact_keys=requested_keys,
    )


def _resolve_recall_policy(
    query: str,
    *,
    base_recall_limit: int,
    requested_relation_depth: int,
) -> dict:
    """Resolve intent with the configured fact-slot vocabulary in both paths."""
    return resolve_intent_recall_policy(
        query,
        config,
        base_recall_limit=base_recall_limit,
        requested_relation_depth=requested_relation_depth,
        fact_slot_registry=_fact_slot_registry(),
    )


def _is_main_recall_bucket(bucket: dict) -> bool:
    """Reject archive/cold material even when a stale index returns its id."""
    if not isinstance(bucket, dict):
        return False
    metadata = bucket.get("metadata", {}) or {}
    if str(metadata.get("type") or "").strip().lower() == "archived":
        return False

    raw_path = str(bucket.get("path") or "").strip()
    archive_dir = str(getattr(bucket_mgr, "archive_dir", "") or "").strip()
    if raw_path and archive_dir:
        try:
            path = os.path.realpath(raw_path)
            archive_root = os.path.realpath(archive_dir)
            if os.path.commonpath([path, archive_root]) == archive_root:
                return False
        except (OSError, ValueError):
            # A malformed/unrelated path is not evidence that an otherwise
            # ordinary in-memory candidate belongs to the cold store.
            pass
    return True


def _passes_nonkeyword_recall_filters(
    bucket: dict,
    *,
    world_filter_set: set | None,
    domain_filter: list[str] | None = None,
    created_after=None,
    created_before=None,
    exclude_core: bool = True,
) -> bool:
    """Mirror BucketManager.search authority filters for side channels."""
    if not _is_main_recall_bucket(bucket):
        return False
    metadata = bucket.get("metadata", {}) or {}
    if exclude_core and (metadata.get("pinned") or metadata.get("protected")):
        return False
    if world_filter_set is not None and not world_matches(
        metadata.get("world", ""), world_filter_set
    ):
        return False
    if created_after is not None or created_before is not None:
        from bucket_manager import _bucket_in_time_range
        if not _bucket_in_time_range(bucket, created_after, created_before):
            return False
    if domain_filter:
        bucket_domains = metadata.get("domain", [])
        if isinstance(bucket_domains, str):
            bucket_domains = [bucket_domains]
        elif not isinstance(bucket_domains, list):
            bucket_domains = []
        requested = {str(value).lower() for value in domain_filter}
        if not ({str(value).lower() for value in bucket_domains} & requested):
            return False
    return True


async def _state_link_recall_candidates(
    seed_buckets,
    *,
    profile: dict,
    world_filter_set: set | None,
    domain_filter: list[str] | None,
    created_after,
    created_before,
    excluded_ids,
    limit: int,
) -> list[dict]:
    """Expand only explicit, reciprocal Z lifecycle links for the asked view."""
    if (
        not profile.get("enabled")
        or profile.get("view") == STATE_VIEW_NEUTRAL
        or not profile.get("fact_keys")
        or limit <= 0
    ):
        return []

    registry = _fact_slot_registry()
    requested_keys = set(profile["fact_keys"])
    excluded = {str(value) for value in (excluded_ids or []) if str(value)}
    seen_sources: set[str] = set()
    results: list[dict] = []

    for source in seed_buckets:
        if len(results) >= limit or not isinstance(source, dict):
            break
        source_id = str(source.get("id") or "")
        if not source_id or source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        source_meta = source.get("metadata", {}) or {}
        source_key = registered_fact_key(source_meta.get("fact_key"), registry)
        source_status = fact_state_label(source, registry)
        if source_key not in requested_keys or not source_status:
            continue

        for target_id in state_link_target_ids(
            source,
            view=str(profile["view"]),
            registry=registry,
        ):
            if len(results) >= limit:
                break
            if target_id in excluded:
                continue
            target = await bucket_mgr.get(target_id)
            if not target or not _passes_nonkeyword_recall_filters(
                target,
                world_filter_set=world_filter_set,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
            ):
                continue
            target_meta = target.get("metadata", {}) or {}
            target_key = registered_fact_key(target_meta.get("fact_key"), registry)
            target_status = fact_state_label(target, registry)
            if target_key != source_key:
                continue
            if profile["view"] == STATE_VIEW_CURRENT and target_status != FACT_STATUS_CURRENT:
                continue
            if (
                profile["view"] == STATE_VIEW_HISTORICAL
                and target_status != FACT_STATUS_HISTORICAL
            ):
                continue
            if profile["view"] == STATE_VIEW_TRANSITION and target_status not in {
                FACT_STATUS_CURRENT,
                FACT_STATUS_HISTORICAL,
            }:
                continue

            target_supersedes = {
                str(value)
                for value in _metadata_list(target_meta.get("supersedes_bucket_ids"))
                if str(value)
            }
            source_supersedes = {
                str(value)
                for value in _metadata_list(source_meta.get("supersedes_bucket_ids"))
                if str(value)
            }
            reciprocal = (
                source_status == FACT_STATUS_HISTORICAL
                and target_status == FACT_STATUS_CURRENT
                and str(source_meta.get("superseded_by_bucket_id") or "") == target_id
                and source_id in target_supersedes
            ) or (
                source_status == FACT_STATUS_CURRENT
                and target_status == FACT_STATUS_HISTORICAL
                and str(target_meta.get("superseded_by_bucket_id") or "") == source_id
                and target_id in source_supersedes
            )
            if not reciprocal:
                continue

            candidate = dict(target)
            candidate["metadata"] = dict(target_meta)
            candidate["_z_state_relation"] = (
                f"supersedes:{source_id}"
                if target_status == FACT_STATUS_CURRENT
                else f"superseded_by:{source_id}"
            )
            candidate["_z_state_via"] = source_id
            results.append(candidate)
            excluded.add(target_id)
    return results


def _z_pair_validation_error(current: dict, historical: dict, fact_key: str) -> str:
    registry = _fact_slot_registry()
    canonical = registered_fact_key(fact_key, registry)
    if canonical is None:
        return "fact_key is no longer registered"
    for label, bucket in (("current", current), ("historical", historical)):
        if not fact_slot_applies_to_bucket(canonical, bucket, registry):
            return f"{label} bucket is outside the registered fact-slot context"
        metadata = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
        existing = str(metadata.get("fact_key") or "").strip().lower()
        if existing and existing != canonical:
            return f"{label} bucket already belongs to a different fact_key"
    return ""


def _relation_recall_neighbors(
    buckets,
    seed_ids,
    *,
    query: str,
    intent: str,
    world_filter,
    domain_filter,
    created_after,
    created_before,
    max_depth: int,
    max_results: int,
    excluded_ids=None,
):
    """Build a bounded, typed Y-axis expansion from already-loaded buckets."""
    candidates = [
        bucket
        for bucket in buckets
        if isinstance(bucket, dict) and _is_main_recall_bucket(bucket)
    ]
    wf_set = {str(value).strip() for value in world_filter} if world_filter is not None else None
    domain_set = {str(value).strip().lower() for value in (domain_filter or [])}

    def eligible(bucket):
        metadata = bucket.get("metadata", {}) or {}
        if wf_set is not None and not world_matches(metadata.get("world", ""), wf_set):
            return False
        if domain_set:
            bucket_domains = {
                str(value).strip().lower()
                for value in _metadata_list(metadata.get("domain", []))
            }
            if not bucket_domains.intersection(domain_set):
                return False
        if created_after is not None or created_before is not None:
            from bucket_manager import _bucket_in_time_range
            if not _bucket_in_time_range(bucket, created_after, created_before):
                return False
        return True

    candidates = [bucket for bucket in candidates if eligible(bucket)]
    candidates = _filter_z_fact_candidates(candidates, query=query, intent=intent)
    allowed_node_ids = {
        str(bucket.get("id"))
        for bucket in candidates
        if bucket.get("id")
    }
    allowed_node_ids.difference_update({
        str(value)
        for value in (excluded_ids or [])
        if str(value)
    })

    relation_cfg = config.get("relation_recall", {}) or {}
    raw_propagation_only = relation_cfg.get("propagation_only", True)
    if isinstance(raw_propagation_only, str):
        propagation_only = raw_propagation_only.strip().lower() in {
            "1", "true", "yes", "on",
        }
    else:
        propagation_only = bool(raw_propagation_only)

    if propagation_only:
        classification = PROPAGATION_RELATION_TYPES
        raw_allowed_types = relation_cfg.get(
            "propagation_types", PROPAGATION_RELATION_TYPES
        )
    else:
        # Exact rollback to the pre-classification behavior.  Keep this branch
        # separate so a legacy allowed_types=[kin, explains] config cannot
        # suppress hard-edge types after propagation mode is enabled.
        classification = SAFE_RELATION_TYPES
        raw_allowed_types = relation_cfg.get(
            "allowed_types", SAFE_RELATION_TYPES
        )
    if isinstance(raw_allowed_types, str):
        raw_allowed_types = [raw_allowed_types]
    elif not isinstance(raw_allowed_types, (list, tuple, set, frozenset)):
        raw_allowed_types = []
    configured_types = {
        str(value).strip()
        for value in raw_allowed_types
    }
    # Unknown/storage-only edge types degrade to non-propagating semantics.
    allowed_types = configured_types.intersection(classification)

    def threshold(name: str, default: float) -> float:
        try:
            value = float(relation_cfg.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if 0.0 <= value <= 1.0 else default

    return expand_relation_graph(
        buckets,
        seed_ids,
        allowed_types=allowed_types,
        max_depth=max_depth,
        max_results=max_results,
        allowed_node_ids=allowed_node_ids,
        hop_min_strength={
            1: threshold("hop1_min_strength", 0.4),
            2: threshold("hop2_min_strength", 0.7),
        },
    )


def _timeline_recall_neighbors(
    buckets,
    seed_ids,
    *,
    query: str,
    intent: str,
    world_filter,
    domain_filter,
    created_after,
    created_before,
    max_results: int,
    excluded_ids=None,
):
    """Build bounded X navigation from displayed primary results."""

    timeline_cfg = config.get("timeline_recall", {}) or {}
    raw_enabled = timeline_cfg.get("enabled", True)
    if isinstance(raw_enabled, str):
        enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(raw_enabled)
    if not enabled or max_results <= 0:
        return []
    try:
        neighbor_window = int(timeline_cfg.get("neighbor_window", 1))
    except (TypeError, ValueError):
        neighbor_window = 1
    neighbor_window = max(0, min(neighbor_window, 4))
    if neighbor_window == 0:
        return []

    candidates = [
        bucket
        for bucket in buckets
        if isinstance(bucket, dict) and _is_main_recall_bucket(bucket)
    ]
    wf_set = (
        {str(value).strip() for value in world_filter}
        if world_filter is not None
        else None
    )
    domain_set = {
        str(value).strip().lower()
        for value in (domain_filter or [])
    }

    def eligible(bucket):
        metadata = bucket.get("metadata", {}) or {}
        if wf_set is not None and not world_matches(
            metadata.get("world", ""),
            wf_set,
        ):
            return False
        if domain_set:
            bucket_domains = {
                str(value).strip().lower()
                for value in _metadata_list(metadata.get("domain", []))
            }
            if not bucket_domains.intersection(domain_set):
                return False
        if created_after is not None or created_before is not None:
            from bucket_manager import _bucket_in_time_range

            if not _bucket_in_time_range(
                bucket,
                created_after,
                created_before,
            ):
                return False
        return True

    candidates = [bucket for bucket in candidates if eligible(bucket)]
    candidates = _filter_z_fact_candidates(
        candidates,
        query=query,
        intent=intent,
    )
    allowed_node_ids = {
        str(bucket.get("id"))
        for bucket in candidates
        if bucket.get("id")
    }
    return timeline_neighbors(
        buckets,
        seed_ids,
        neighbor_window=neighbor_window,
        max_results=max_results,
        allowed_node_ids=allowed_node_ids,
        excluded_ids=excluded_ids,
    )


def _review_gate(name: str) -> bool:
    """Read config.review_gate.<name>; fail safe when omitted."""
    return bool((config.get("review_gate", {}) or {}).get(name, True))


def _recall_prefix(
    bucket_id: str,
    role: str,
    layer: str,
    *,
    marker: str = "",
    relation: str = "",
    bucket: dict | None = None,
    state_profile: dict | None = None,
) -> str:
    """Prefix recall snippets; association is always explicit supporting evidence."""
    roles_enabled = (config.get("recall_evidence_roles", {}) or {}).get("enabled", False)
    state_label = ""
    operational_label = {}
    if (
        state_profile
        and state_profile.get("enabled")
        and state_profile.get("evidence_labels")
        and state_profile.get("view") != STATE_VIEW_NEUTRAL
        and state_profile.get("fact_keys")
        and bucket
    ):
        state_label = fact_state_label(bucket, _fact_slot_registry())
    if state_profile and bucket:
        operational_label = operational_validity_label(
            bucket,
            view=str(
                state_profile.get("operational_view")
                or OPERATIONAL_VIEW_NEUTRAL
            ),
        )
    if roles_enabled or role != "main" or state_label or operational_label:
        parts = [f"[role:{role}]", f"[layer:{layer}]"]
        if role == "state":
            parts.append("[authority:state_evidence]")
        elif role != "main":
            parts.append("[authority:supporting_only]")
        if state_label:
            parts.extend((
                f"[memory_state:{state_label}]",
                f"[query_state_view:{state_profile['view']}]",
            ))
        if operational_label:
            validity_state = str(operational_label.get("state") or "unknown")
            parts.append(f"[validity:{validity_state}]")
            if validity_state == "current":
                parts.append("[authority:current_status]")
            elif validity_state == "historical":
                parts.append("[authority:historical_status]")
            else:
                parts.append("[authority:not_current_status]")
            for field in ("valid_at", "invalid_at", "expired_at"):
                value = str(operational_label.get(field) or "").strip()
                if value:
                    parts.append(f"[{field}:{value.replace(' ', 'T')}]")
        if relation:
            parts.append(f"[relation:{relation}]")
        parts.append(f"[bucket_id:{bucket_id}]")
        return " ".join(parts)
    return f"{marker} [bucket_id:{bucket_id}]" if marker else f"[bucket_id:{bucket_id}]"

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
@asynccontextmanager
async def _server_lifespan(_server):
    """Own process-wide background tasks without delaying server readiness."""
    await pg_mirror_worker.start()
    await _start_briefing_cache_refresh()
    try:
        yield {}
    finally:
        await _stop_briefing_cache_refresh()
        await pg_mirror_worker.stop()


mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
    lifespan=_server_lifespan,
)


BREATH_RECALL_POOL_SIZE = 20
BREATH_DEFAULT_MAX_RESULTS = 8
BREATH_DEFAULT_MAX_TOKENS = 6000
# 精排过滤器的输出预算。原值 200 —— 对不吐思考链的 DeepSeek 够用，
# 但换到会先推理的模型（2026-08-20 朝灯把精排换成 apiroute 上的
# gemini-3.7-flash）之后，隐藏推理把 200 全吃光：finish_reason="length"、
# completion_tokens=196 而 content 只剩 3~19 字符，解析必然失败、
# 每次都回退 stub —— 当天 22:11-23:33 连续 12 次无效 / 21 次回退，无一幸免。
# 实测四变体（.work/probe_rerank_fix.py，真实打 API）：
#   mt=200 失败 · mt=200+extra_body{"thinking":"disabled"} 仍失败（gemini 不认
#   这个 DeepSeek 扩展参数，输出 0 字符）· mt=2000 通过 · mt=2000+关思考 通过但多烧 token。
# 结论：靠关思考不行，只能给够预算。这里不按域名/模型名猜，直接给足。
DS_FILTER_MAX_TOKENS = 2000
PULSE_NAV_SUMMARY_CHARS = 110
MCP_IMAGE_MAX_ITEMS = 3
MCP_IMAGE_MAX_BYTES = 900_000
SESSION_SURFACE_DIRNAME = ".session_surface"


def _bounded_env_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


SESSION_RECALL_TTL_SECONDS = _bounded_env_int(
    "OMBRE_SESSION_RECALL_TTL_SECONDS",
    2 * 86400,
    60,
)
SESSION_RECALL_MAX_KEYS = _bounded_env_int(
    "OMBRE_SESSION_RECALL_MAX_KEYS",
    1024,
    1,
)
IMAGE_MARKDOWN_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^\s)]+)\)")


def _bucket_icon(meta: dict) -> str:
    if meta.get("pinned") or meta.get("protected"):
        return "📌"
    if meta.get("type") == "permanent":
        return "📦"
    if meta.get("type") == "feel":
        return "🫧"
    if meta.get("type") == "archived":
        return "🗄️"
    if meta.get("resolved", False):
        return "✅"
    return "💭"


def _collapse_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _clip_text(text: str, max_chars: int) -> str:
    text = _collapse_ws(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _strip_markdown_images(text: str) -> str:
    return IMAGE_MARKDOWN_RE.sub("", text or "")


def _bucket_navigator_summary(bucket: dict, max_chars: int = PULSE_NAV_SUMMARY_CHARS) -> str:
    raw = bucket.get("content", "") or ""
    summary = ""

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or "").strip()
            if not summary:
                facts = parsed.get("core_facts") or []
                if isinstance(facts, list) and facts:
                    summary = str(facts[0] or "").strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    if not summary:
        plain = strip_wikilinks(_strip_markdown_images(raw))
        for line in plain.splitlines():
            if line.strip():
                summary = line.strip()
                break

    return _clip_text(summary or "无摘要，inspect 查看原文", max_chars)


def _format_pulse_line(bucket: dict, score: float, full: bool = False) -> str:
    meta = bucket.get("metadata", {})
    icon = _bucket_icon(meta)
    domains = ",".join(meta.get("domain", []) or [])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.3)
    resolved_tag = " [已解决]" if meta.get("resolved", False) else ""

    if full:
        return (
            f"{icon} [{meta.get('name', bucket['id'])}]{resolved_tag} "
            f"bucket_id:{bucket['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []) or [])}"
        )

    summary = _bucket_navigator_summary(bucket)
    return (
        f"{icon} [{meta.get('name', bucket['id'])}]{resolved_tag} "
        f"bucket_id:{bucket['id']} "
        f"主题:{domains or '未分类'} "
        f"重要:{meta.get('importance', '?')} "
        f"权重:{score:.2f} "
        f"摘要:{summary} "
        f"inspect:{bucket['id']}"
    )


def _session_recall_history() -> JsonFileRecallHistory:
    return JsonFileRecallHistory(
        os.path.join(config["buckets_dir"], SESSION_SURFACE_DIRNAME),
        ttl_seconds=SESSION_RECALL_TTL_SECONDS,
        max_keys_per_session=SESSION_RECALL_MAX_KEYS,
    )


def _session_bucket_key(bucket_id: str) -> str:
    return recall_identity("curated", str(bucket_id))


def _session_seen_bucket_ids(buckets: list[dict], session_id: str) -> set[str]:
    if not session_id or not session_id.strip() or not buckets:
        return set()
    keyed = {
        _session_bucket_key(str(bucket.get("id"))): str(bucket.get("id"))
        for bucket in buckets
        if isinstance(bucket, dict) and bucket.get("id")
    }
    try:
        seen = _session_recall_history().seen(session_id.strip(), keyed)
    except Exception as exc:
        logger.warning("Session recall history read failed open: %s", type(exc).__name__)
        return set()
    return (
        {keyed[key] for key in seen if key in keyed}
        | _load_session_seen_ids(session_id)
    )


def _load_session_seen_ids(session_id: str) -> set[str]:
    """Read only pre-53a4aaa plaintext state during the format migration."""
    if not session_id or not session_id.strip():
        return set()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id.strip())[:80]
    safe_id = safe_id.strip("._-") or "default"
    path = os.path.join(
        config["buckets_dir"],
        SESSION_SURFACE_DIRNAME,
        f"{safe_id}.json",
    )
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return {str(value) for value in data if value}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return set()


def _remember_session_seen_ids(session_id: str, bucket_ids: list[str]) -> None:
    if not session_id or not session_id.strip() or not bucket_ids:
        return
    try:
        with shared_acceptance_write_guard():
            keys = [_session_bucket_key(str(value)) for value in bucket_ids if value]
            _session_recall_history().mark(session_id.strip(), keys)
    except RawIngestGuardError:
        # A strict acceptance window suppresses read-side persistence only;
        # the memory response itself remains available.
        return
    except OSError as e:
        logger.warning(f"Session surface dedup write failed / 会话去重写入失败: {e}")


def _filter_session_seen(buckets: list[dict], session_id: str) -> list[dict]:
    seen = _session_seen_bucket_ids(buckets, session_id)
    if not seen:
        return buckets
    return [bucket for bucket in buckets if str(bucket.get("id")) not in seen]


def _dedupe_recall_content(buckets: list[dict]) -> tuple[list[dict], int, int]:
    """Port LMC-5's same-turn content dedup without duplicate score inflation."""
    output: list[dict] = []
    by_fingerprint: dict[str, int] = {}
    suppressed = 0
    errors = 0
    for bucket in buckets:
        try:
            fingerprint = default_content_fingerprint(str(bucket.get("content") or ""))
        except Exception:
            fingerprint = None
            errors += 1
        if not fingerprint:
            output.append(bucket)
            continue
        existing_index = by_fingerprint.get(fingerprint)
        if existing_index is None:
            by_fingerprint[fingerprint] = len(output)
            output.append(bucket)
            continue
        suppressed += 1
        winner = output[existing_index]
        metadata = winner.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["content_duplicates_merged"] = (
                int(metadata.get("content_duplicates_merged") or 0) + 1
            )
    return output, suppressed, errors


def _get_sensory_engine() -> SensoryEngine:
    """Keep tests/config overrides from pinning the engine to an old buckets_dir."""
    global sensory_engine
    buckets_dir = config["buckets_dir"]
    if getattr(sensory_engine, "buckets_dir", None) != buckets_dir:
        sensory_engine = SensoryEngine(buckets_dir)
    return sensory_engine


def _append_body_state_block(
    text: str,
    buckets: list[dict],
    session_id: str = "",
    include_body_state: bool = True,
    reset_body_state: bool = False,
) -> str:
    """Append generated body-state data; bucket text never becomes instructions."""
    if reset_body_state:
        _get_sensory_engine().reset_state()
    if not include_body_state:
        return text
    try:
        seen = _session_seen_bucket_ids(buckets, session_id) if session_id else set()
        result = _get_sensory_engine().stimulate_from_buckets(
            buckets,
            seen_ids=seen,
        )
        if result.triggered_bucket_ids:
            _remember_session_seen_ids(session_id, result.triggered_bucket_ids)
        block = format_body_state_block(result)
    except Exception as e:
        logger.warning(f"Sensory body-state update failed / 感官状态更新失败: {e}")
        block = ""
    if not block:
        return text
    return f"{text}\n\n{block}" if text else block


def _ds_gate_enabled(mode: str) -> bool:
    """DeepSeek 语义门控开关。默认关；仅对 OMBRE_DS_FILTER_MODES 列出的 mode 生效（默认只 search）。"""
    flag = os.getenv("OMBRE_DS_FILTER_ENABLED", "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    modes = os.getenv("OMBRE_DS_FILTER_MODES", "search")
    allowed = {m.strip() for m in modes.split(",") if m.strip()}
    return mode in allowed


def _ds_gate_timeout() -> float:
    try:
        return float(os.getenv("OMBRE_DS_FILTER_TIMEOUT", "8"))
    except ValueError:
        return 8.0


def _normalize_anchor_recall_policy(value: str) -> str:
    policy = str(value or "search").strip().lower()
    return policy if policy in ANCHOR_RECALL_POLICIES else "search"


def _anchor_quality_gate_enabled(policy: str) -> bool:
    """Return whether the vendored Anchor threshold applies to this path."""
    flag = os.getenv("OMBRE_ANCHOR_QUALITY_GATE_ENABLED", "1").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    configured = os.getenv(
        "OMBRE_ANCHOR_QUALITY_GATE_POLICIES",
        "conversation,reflex",
    )
    enabled_policies = {
        value.strip().lower()
        for value in configured.split(",")
        if value.strip()
    }
    return policy in enabled_policies


def _anchor_adapted_relevance_score(bucket: dict) -> float | None:
    """Map Ombre's absolute query evidence onto Anchor's score scale.

    Anchor ``recall_v2._anchor_item`` gives query similarity a 0.45 weight.
    Ombre's fused RRF score is rank-relative and therefore cannot be compared
    with Anchor's absolute 0.25 conversation threshold.  Literal relevance
    (0..100) and original-vector similarity (0..1) are absolute query signals,
    so normalising those and applying the same 0.45 weight creates a conservative
    lower bound on the vendored scale.  Importance, decay and E resonance are
    deliberately excluded: none of them proves that this memory answers the
    current query.
    """
    similarities: list[float] = []

    literal = bucket.get("_literal_relevance_score")
    if isinstance(literal, (int, float)):
        similarities.append(max(0.0, min(1.0, float(literal) / 100.0)))

    vector = bucket.get("_original_vector_relevance_score")
    if isinstance(vector, (int, float)):
        similarities.append(max(0.0, min(1.0, float(vector))))

    # A current, validated entity link means the query explicitly named the
    # entity.  Treat that as anchored query evidence, not a ranking bonus.
    if bucket.get("entity_match"):
        similarities.append(1.0)

    if not similarities:
        return None
    return round(0.45 * max(similarities), 6)


def _filter_anchor_policy_candidates(
    candidates: list[dict],
    policy: str,
) -> list[dict]:
    """Apply Anchor's policy threshold without changing search/manual recall.

    Missing score evidence is kept (fail-open) so an adapter/config regression
    cannot silently erase the existing recall path.  A genuine score below the
    selected Anchor threshold is allowed to disappear for conversation/reflex,
    which is the upstream legal-empty contract.
    """
    recall_policy = _normalize_anchor_recall_policy(policy)
    if not candidates or not _anchor_quality_gate_enabled(recall_policy):
        return candidates

    preset = ANCHOR_RECALL_POLICIES.get(recall_policy)
    if not isinstance(preset, dict):
        return candidates
    try:
        threshold = float(preset["min_score"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Anchor quality gate missing threshold; keeping candidates")
        return candidates

    kept: list[dict] = []
    scored = 0
    for bucket in candidates:
        score = _anchor_adapted_relevance_score(bucket)
        if score is None:
            kept.append(bucket)
            continue
        scored += 1
        bucket["_anchor_adapted_relevance_score"] = score
        if score >= threshold:
            kept.append(bucket)

    if scored == 0:
        logger.warning("Anchor quality gate had no absolute scores; keeping candidates")
        return candidates
    logger.info(
        "Anchor quality gate policy=%s threshold=%.3f input=%d scored=%d kept=%d",
        recall_policy,
        threshold,
        len(candidates),
        scored,
        len(kept),
    )
    return kept


def _parse_ds_keep_indices(raw: str, n: int) -> list[int] | None:
    """解析 DeepSeek 返回的 ``keep``。

    ``[]`` 是模型明确给出的合法判空；``None`` 才表示协议/解析失败。
    这一区分复用 Anchor ``allow_empty`` 合同：对话召回可以安静，模型
    故障仍必须回退既有候选，不能把故障冒充成“没有记忆”。
    """
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, IndexError):
        return None
    arr = data.get("keep") if isinstance(data, dict) else data
    if not isinstance(arr, list):
        return None
    out: list[int] = []
    for x in arr:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            out.append(i)
    if arr and not out:
        # 只有字面上的空数组才是“合法判空”；非空但全越界/非法属于坏协议。
        return None
    return out


def _exact_retrieval_key_ids(query: str, buckets: list[dict]) -> set[str]:
    """Return buckets whose curated retrieval key occurs verbatim in query."""
    query_text = str(query or "").casefold()
    if not query_text:
        return set()

    matched: set[str] = set()
    for bucket in buckets:
        metadata = bucket.get("metadata", {}) or {}
        candidates = metadata.get("retrieval_keys", [])
        if isinstance(candidates, str):
            candidates = [candidates]
        elif not isinstance(candidates, (list, tuple, set)):
            candidates = []
        keys = literal_retrieval_keys(
            str(bucket.get("content") or ""),
            candidates,
        )
        if not any(str(key).casefold() in query_text for key in keys):
            continue
        bucket_id = str(bucket.get("id") or "").strip()
        if bucket_id:
            matched.add(bucket_id)
    return matched


def _cap_candidates_preserving_forced(
    candidates: list[dict],
    force_keep_ids: set[str],
    max_results: int,
) -> list[dict]:
    """Keep forced rows in order while making them consume the normal cap."""
    if max_results <= 0:
        return []
    forced_count = sum(
        1 for bucket in candidates if bucket.get("id") in force_keep_ids
    )
    ordinary_budget = max(0, max_results - forced_count)
    selected: list[dict] = []
    for bucket in candidates:
        if bucket.get("id") in force_keep_ids:
            selected.append(bucket)
        elif ordinary_budget > 0:
            selected.append(bucket)
            ordinary_budget -= 1
    return selected


async def _ds_semantic_select(
    query: str,
    buckets: list[dict],
    keep: set[str],
    max_results: int,
) -> list[dict]:
    """用 DeepSeek 判断每条候选是否与 query 语义相关；纯减法（只剔噪、不重排不外拉），forced 恒留。"""
    client = getattr(dehydrator, "client", None)
    if client is None:
        raise RuntimeError("no DeepSeek client configured")
    lines = []
    for i, b in enumerate(buckets):
        name = redact_embedding_input((b.get("metadata", {}) or {}).get("name") or b.get("id", ""))
        snippet = redact_embedding_input((b.get("content") or "").strip().replace("\n", " "))[:200]
        lines.append(f"[{i}] {name}: {snippet}")
    sys_prompt = (
        "你是记忆召回的相关性过滤器。给定用户查询和一组候选记忆条目，"
        "判断每条是否与查询语义相关、值得进入上下文。"
        '只返回 JSON：{"keep": [相关条目的序号整数数组]}，不要解释。'
        "宁可多留也别漏掉明显相关的；只剔除与查询确实无关的。"
    )
    user_prompt = f"查询：{redact_embedding_input(query)}\n\n候选：\n" + "\n".join(lines)
    resp = await client.chat.completions.create(
        model=getattr(dehydrator, "model", "deepseek-chat"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=DS_FILTER_MAX_TOKENS,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content if resp.choices else ""
    idxs = _parse_ds_keep_indices(raw, len(buckets))
    if idxs is None:
        logger.error(
            "DS filter received invalid DeepSeek response raw_chars=%d "
            "raw_sha256=%s response=%s",
            len(raw),
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            _safe_chat_completion_diagnostics(resp),
        )
        raise ValueError("invalid DeepSeek recall selection payload")
    keep_idx = set(idxs)
    selected = [
        b for i, b in enumerate(buckets)
        if i in keep_idx or b.get("id") in keep
    ]
    return _cap_candidates_preserving_forced(selected, keep, max_results)


async def _ds_filter_candidates(
    query: str,
    candidates: list[dict],
    *,
    mode: str,
    max_results: int,
    force_keep_ids: set[str] = None,
    allow_empty: bool = False,
) -> list[dict]:
    """
    召回候选的注入裁剪 + 可选 DeepSeek 语义门控。

    默认行为（门控关，PR-1 语义）：保序 + 保留 forced IDs + 限到 max_results，不调 LLM。
    门控开（OMBRE_DS_FILTER_ENABLED 且 mode 命中且 query 非空）：在已裁剪集合上跑 DeepSeek
    相关性过滤，纯减法剔噪。超时/出错/解析失败一律回退裁剪集合；只有调用方
    明确采用 Anchor 对话策略并传入 ``allow_empty=True`` 时，合法 ``keep: []``
    才会安静返回空。
    """
    if max_results <= 0:
        return []
    keep = force_keep_ids or set()
    capped = _cap_candidates_preserving_forced(candidates, keep, max_results)

    if not _ds_gate_enabled(mode) or not query or not capped:
        logger.debug(
            "DS filter stub mode=%s query=%r input=%d output=%d",
            mode,
            query[:80] if query else "",
            len(candidates),
            len(capped),
        )
        return capped

    # The gate is subtractive only.  With one non-empty result required, a
    # singleton can never change: keeping it returns ``capped`` and rejecting
    # it also falls back to ``capped`` below.  Likewise, forced candidates can
    # never be removed.  Avoid paying for a model decision whose result is
    # already determined locally.
    if (len(capped) == 1 and not allow_empty) or all(
        bucket.get("id") in keep for bucket in capped
    ):
        logger.debug(
            "DS filter deterministic no-op mode=%s query=%r capped=%d",
            mode,
            query[:80],
            len(capped),
        )
        return capped

    try:
        kept = await asyncio.wait_for(
            _ds_semantic_select(query, capped, keep, max_results),
            timeout=_ds_gate_timeout(),
        )
    except Exception as e:
        logger.warning(
            "DS filter fell back to stub / 门控回退裁剪集合 (%s): %s",
            type(e).__name__, e,
        )
        return capped

    result = kept if kept or allow_empty else capped
    logger.info(
        "DS filter mode=%s query=%r input=%d capped=%d kept=%d",
        mode, query[:80], len(candidates), len(capped), len(result),
    )
    return result


def _extract_markdown_images(text: str) -> list[tuple[str, str]]:
    return [(alt.strip(), url.strip()) for alt, url in IMAGE_MARKDOWN_RE.findall(text or "")]


def _bucket_allows_mcp_image(bucket: dict) -> bool:
    meta = bucket.get("metadata", {}) or {}
    if meta.get("pinned") or meta.get("protected"):
        return True
    if meta.get("type") == "feel":
        # feel 桶带图 = 私密锚点（胸口照/绿月夜那类），本就该让哥哥看见；
        # 无图的 feel 不受影响（_collect_mcp_images 只对真有图的桶出图）
        return True
    try:
        if int(meta.get("importance", 0)) >= 8:
            return True
    except (TypeError, ValueError):
        pass
    return _is_anchor_bucket(bucket)


def _is_anchor_bucket(bucket: dict) -> bool:
    """anchor 桶：带 anchor/锚/mcp-image 标签。用于出图优先级排序。"""
    meta = bucket.get("metadata", {})
    tags = [str(t).lower() for t in (meta.get("tags", []) or [])]
    return any("anchor" in t or "锚" in t or "mcp-image" in t for t in tags)


def _is_r2_image_url(url: str) -> bool:
    if not url:
        return False
    public_url = getattr(r2_storage, "public_url", "") or ""
    if public_url and url.startswith(public_url.rstrip("/") + "/"):
        return True
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host.endswith(".r2.dev") or host.endswith(".r2.cloudflarestorage.com")


def _mime_from_url_or_header(url: str, content_type: str = "") -> str:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith("image/"):
        return mime
    guessed, _ = mimetypes.guess_type(url)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "application/octet-stream"


async def _fetch_mcp_image_content(bucket: dict, url: str) -> ImageContent | None:
    if not _is_r2_image_url(url):
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as e:
        logger.warning(f"MCP image fetch failed / MCP 图片拉取失败: {url}: {e}")
        return None

    blob = resp.content or b""
    if not blob or len(blob) > MCP_IMAGE_MAX_BYTES:
        logger.info(
            "MCP image skipped size=%d max=%d url=%s",
            len(blob),
            MCP_IMAGE_MAX_BYTES,
            url,
        )
        return None

    mime = _mime_from_url_or_header(url, resp.headers.get("content-type", ""))
    if not mime.startswith("image/"):
        return None
    data = base64.b64encode(blob).decode("ascii")
    return ImageContent(
        type="image",
        data=data,
        mimeType=mime,
        _meta={"bucket_id": bucket.get("id"), "source_url": url},
    )


async def _collect_mcp_images(buckets: list[dict]) -> list[ImageContent]:
    images: list[ImageContent] = []
    seen_urls: set[str] = set()
    # anchor 桶优先：MAX_ITEMS 截断时先保住锚点图（胸口照/绿月夜等），稳定排序保留原序
    ordered = sorted(buckets, key=lambda b: 0 if _is_anchor_bucket(b) else 1)
    for bucket in ordered:
        if len(images) >= MCP_IMAGE_MAX_ITEMS:
            break
        if not _bucket_allows_mcp_image(bucket):
            continue
        for _alt, url in _extract_markdown_images(bucket.get("content", "")):
            if len(images) >= MCP_IMAGE_MAX_ITEMS:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)
            image = await _fetch_mcp_image_content(bucket, url)
            if image:
                images.append(image)
    return images


async def _tool_result_with_optional_images(
    text: str,
    buckets: list[dict],
    include_images: bool,
) -> str | list[TextContent | ImageContent]:
    if not include_images:
        return text
    images = await _collect_mcp_images(buckets)
    if not images:
        return text
    return [TextContent(type="text", text=text), *images]


_E_AXIS_ROWS_CACHE: dict[str, object] = {"token": None, "cfg": None, "rows": {}}


async def _e_axis_rows_cached(e_recall_cfg) -> dict:
    """Serve E-axis grouping from a snapshot-keyed cache of its tiny result.

    E projection only needs the few primary-authored buckets, but list_all()
    deep-copies the whole library per call; that copy dominated query_prep
    (4-5s per beat at ~12k buckets).  The cache key reuses the directory
    snapshot list_all() itself invalidates on, so any bucket write refreshes
    this view naturally and E behaviour stays byte-identical.
    """
    token_fn = getattr(bucket_mgr, "list_all_snapshot_token", None)
    cfg_key = (e_recall_cfg.enabled, e_recall_cfg.min_confidence)
    token = await token_fn(include_archive=False) if callable(token_fn) else None
    cache = _E_AXIS_ROWS_CACHE
    if token is not None and cache["token"] == token and cache["cfg"] == cfg_key:
        return cache["rows"]  # read-only rows; projection never mutates them
    rows = group_primary_authored_buckets(
        await bucket_mgr.list_all(include_archive=False),
        e_recall_cfg,
    )
    if token is not None:
        cache["token"] = token
        cache["cfg"] = cfg_key
        cache["rows"] = rows
    return rows


def _recall_dehydrate_async_enabled() -> bool:
    cfg = config.get("dehydration", {}) or {}
    if cfg.get("recall_async_backfill_enabled") is False:
        return False
    flag = os.environ.get("OMBRE_RECALL_DEHYDRATE_ASYNC", "1").strip().lower()
    return flag not in {"0", "false", "off"}


_DEHYDRATE_BACKFILL_PENDING: set[str] = set()
_DEHYDRATE_BACKFILL_SEM = asyncio.Semaphore(2)


async def _extend_e_axis_cache_after_summary_write(token_pre) -> None:
    """Re-key the E-axis cache after a summary-only frontmatter write.

    cache_recall_dehydration touches only ``dehydrated_summary`` /
    ``dehydrated_content_hash`` — fields the E grouping never reads — yet the
    write moves the directory snapshot and would evict the cache, forcing the
    next beat back onto a full-library rescan (the two perf fixes fighting
    each other).  Extension is gated on ``token_pre`` (snapshot taken before
    the write) still matching the cache: if any real bucket write slipped in
    before ours, the tokens disagree and we leave eviction to do its job.
    The remaining race (another write between our write and the token read
    below) can only delay one freshly-authored E row until the next write —
    it cannot corrupt existing rows, because summary writes never change the
    fields the grouping reads.
    """
    cache = _E_AXIS_ROWS_CACHE
    if token_pre is None or cache["token"] != token_pre:
        return
    token_fn = getattr(bucket_mgr, "list_all_snapshot_token", None)
    if not callable(token_fn):
        return
    try:
        cache["token"] = await token_fn(include_archive=False)
    except Exception as exc:
        logger.warning(
            "E-axis cache token extension failed; next beat rebuilds: %s",
            type(exc).__name__,
        )
        cache["token"] = None


def _schedule_recall_dehydration_backfill(bucket_id: str, content: str, body_hash: str) -> None:
    """Compute the LLM summary off the recall path and persist it for next beat."""
    if not bucket_id or bucket_id in _DEHYDRATE_BACKFILL_PENDING:
        return
    _DEHYDRATE_BACKFILL_PENDING.add(bucket_id)

    async def _run() -> None:
        try:
            async with _DEHYDRATE_BACKFILL_SEM:
                with_source = getattr(dehydrator, "dehydrate_with_source", None)
                if not callable(with_source):
                    return
                raw_summary, _source = await with_source(content, None, write_cache=False)
                if not isinstance(raw_summary, str) or len(raw_summary.strip()) < 10:
                    return
                writer = getattr(bucket_mgr, "cache_recall_dehydration", None)
                if callable(writer):
                    token_fn = getattr(bucket_mgr, "list_all_snapshot_token", None)
                    token_pre = (
                        await token_fn(include_archive=False)
                        if callable(token_fn)
                        else None
                    )
                    persisted = await writer(
                        bucket_id,
                        expected_content_hash=body_hash,
                        summary=raw_summary,
                    )
                    if persisted:
                        await _extend_e_axis_cache_after_summary_write(token_pre)
        except Exception as exc:
            logger.warning(
                "Async recall dehydration backfill failed for %s: %s",
                bucket_id,
                type(exc).__name__,
            )
        finally:
            _DEHYDRATE_BACKFILL_PENDING.discard(bucket_id)

    asyncio.create_task(_run())


def _frontmatter_dehydration_cache_enabled() -> bool:
    cfg = config.get("dehydration", {}) or {}
    return cfg.get("recall_frontmatter_cache_enabled", True) is not False


async def _dehydrate_for_recall(
    content: str,
    metadata: dict,
    *,
    bucket: dict | None = None,
    allow_async_fallback: bool = False,
) -> str:
    """Render recall text and persist only its derived bucket summary.

    ``dehydrated_summary`` is valid while the exact Markdown body hash is
    unchanged.  The write path is isolated from ``last_active`` and other
    factual metadata, so caching cannot heat a bucket or alter ranking.
    """
    cache_enabled = _frontmatter_dehydration_cache_enabled()
    bucket_metadata = (
        bucket.get("metadata", {})
        if isinstance(bucket, dict) and isinstance(bucket.get("metadata"), dict)
        else {}
    )
    raw_body = str(bucket.get("content") or "") if isinstance(bucket, dict) else ""
    body_hash = hashlib.sha256(raw_body.encode("utf-8")).hexdigest() if bucket else ""
    stored_summary = bucket_metadata.get("dehydrated_summary")
    if (
        cache_enabled
        and isinstance(stored_summary, str)
        and len(stored_summary.strip()) >= 10
        and bucket_metadata.get("dehydrated_content_hash") == body_hash
    ):
        record_recall_dehydration("frontmatter_hits")
        formatter = getattr(dehydrator, "format_dehydration_summary", None)
        if callable(formatter):
            return formatter(stored_summary.strip(), metadata)
        return stored_summary.strip()

    if cache_enabled and bucket and allow_async_fallback and _recall_dehydrate_async_enabled():
        # Recall beats must not wait on a live LLM summary.  Serve a truncated
        # body now, push the real dehydration to a background task that writes
        # the frontmatter cache, and let the next beat hit it.  Quality dips
        # once per bucket, latency spike disappears every time.
        fallback_id = str(bucket.get("id") or "")
        if fallback_id:
            _schedule_recall_dehydration_backfill(fallback_id, content, body_hash)
            record_recall_dehydration("passthrough_async")
            squashed = " ".join(content.split())
            fallback = squashed[:300] + ("…" if len(squashed) > 300 else "")
            formatter = getattr(dehydrator, "format_dehydration_summary", None)
            if callable(formatter):
                return formatter(fallback, metadata)
            return fallback

    with_source = getattr(dehydrator, "dehydrate_with_source", None)
    if callable(with_source):
        raw_summary, source = await with_source(
            content,
            None,
            write_cache=False,
        )
        formatter = getattr(dehydrator, "format_dehydration_summary", None)
        rendered = (
            formatter(raw_summary, metadata)
            if callable(formatter)
            else raw_summary
        )
    else:
        # Test doubles and older rollback implementations keep the original
        # dehydrate signature.  Production always uses the source-aware path.
        rendered = await dehydrator.dehydrate(
            content,
            metadata,
            write_cache=False,
        )
        raw_summary = rendered
        source = "computed"

    if source == "cached":
        record_recall_dehydration("backfilled")
    elif source == "computed":
        record_recall_dehydration("computed")
    else:
        record_recall_dehydration("passthrough")

    if cache_enabled and bucket and source != "passthrough":
        writer = getattr(bucket_mgr, "cache_recall_dehydration", None)
        persisted = False
        if callable(writer):
            try:
                token_fn = getattr(bucket_mgr, "list_all_snapshot_token", None)
                token_pre = (
                    await token_fn(include_archive=False)
                    if callable(token_fn)
                    else None
                )
                persisted = await writer(
                    str(bucket.get("id") or ""),
                    expected_content_hash=body_hash,
                    summary=raw_summary,
                )
                if persisted:
                    # Same summary-only write as the async backfill path: keep
                    # the E-axis cache alive instead of letting this evict it.
                    await _extend_e_axis_cache_after_summary_write(token_pre)
            except Exception as exc:
                logger.warning(
                    "Recall summary frontmatter write failed for %s: %s",
                    bucket.get("id"),
                    type(exc).__name__,
                )
        if not persisted:
            record_recall_dehydration("persist_failed")
    return rendered


def _local_partial_recall_text(
    candidates: list[dict],
    *,
    max_results: int,
    max_tokens: int,
    state_profile: dict,
) -> str:
    """Render already-approved candidates without another external API call."""
    rendered: list[str] = []
    token_used = 0
    for bucket in candidates[:max_results]:
        bucket_id = str(bucket.get("id") or "")
        if not bucket_id:
            continue
        content = strip_wikilinks(str(bucket.get("content") or "")).strip()
        excerpt = ""
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            summary = payload.get("summary")
            if isinstance(summary, str) and summary.strip():
                excerpt = summary.strip()
            else:
                facts = payload.get("core_facts")
                if isinstance(facts, list):
                    excerpt = "；".join(
                        str(item).strip()
                        for item in facts[:2]
                        if str(item).strip()
                    )
        if not excerpt:
            excerpt = re.sub(r"\s+", " ", content)[:800].strip()
        excerpt = redact_text(excerpt)
        if not excerpt:
            continue
        prefix = _recall_prefix(
            bucket_id,
            "main",
            "curated_rrf_partial",
            bucket=bucket,
            state_profile=state_profile,
        )
        line = f"{prefix} {excerpt}"
        line_tokens = count_tokens_approx(line)
        if token_used + line_tokens > max_tokens:
            break
        rendered.append(line)
        token_used += line_tokens
    return "\n---\n".join(rendered)


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # pinned
        pinned = [b for b in all_buckets if b["metadata"].get("pinned") or b["metadata"].get("protected")]
        # top 2 unresolved by score
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not b["metadata"].get("pinned")
                      and not b["metadata"].get("protected")]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"}, bucket=b)
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)

        # --- Feel buckets: emotional sediment, surface right after pinned ---
        # --- feel 桶:情感沉淀,紧跟核心准则浮现(独立池) ---
        feel_seen = {b["id"] for b in pinned}
        for b in _surface_feel_pool(all_buckets, feel_seen):
            summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"}, bucket=b)
            parts.append(f"💧 [情感沉淀] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        for b in candidates:
            if token_budget <= 0:
                break
            summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"}, bucket=b)
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            token_budget -= summary_tokens

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse(redact_text("[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts)))
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
def _sample_dream_material(candidates: list[dict], n: int = 0) -> list[dict]:
    """梦的「材料混合器」。

    旧版只取最近 10 条 → 梦永远像「昨日日报改写」。这里把材料混起来：
      最近记忆(40%) / 未解决残渣·偏老(25%) / 感官锚点(15%) / 随机小碎片(余量补满)，
    最后**打乱顺序**——梦会跳跃、不按时间线走。n<=0 时随机 8~14 条（梦有长有短）。

    纯函数、只读、不写库。配额各至少 1 条；候选不足时尽力填、不报错。
    TODO(#4 落库后): 掺入「上一两个梦的残影」(dream_events.jsonl 的 source/正文)，
    让梦能接住前一晚的影子，而不是每晚从零。
    """
    if not candidates:
        return []
    if n <= 0:
        n = random.randint(8, 14)

    def _created(b):
        return event_at_from_metadata(b["metadata"]) or ""

    by_recent = sorted(candidates, key=_created, reverse=True)
    unresolved_old = sorted(
        [b for b in candidates if not b["metadata"].get("resolved", False)],
        key=_created,
    )
    anchors = [b for b in candidates if _is_anchor_bucket(b)]
    fragments = list(candidates)
    random.shuffle(fragments)

    q_recent = max(1, round(n * 0.40))
    q_unresolved = max(1, round(n * 0.25))
    q_anchor = max(1, round(n * 0.15)) if anchors else 0

    picks: list[dict] = []
    seen: set = set()

    def _take(pool, k):
        for b in pool:
            if len(picks) >= n or k <= 0:
                break
            bid = b["id"]
            if bid in seen:
                continue
            seen.add(bid)
            picks.append(b)
            k -= 1

    _take(by_recent, q_recent)
    _take(unresolved_old, q_unresolved)
    _take(anchors, q_anchor)
    _take(fragments, n - len(picks))   # 余量用随机碎片补满（含很老的普通桶 → 无厘头接点）
    random.shuffle(picks)              # 关键：打乱时间线，梦不是日报
    return picks


@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        # 可选 ?n=：让 claude-twin 的 DreamProfile 调长短梦（长梦多取、短梦少取）。
        # request 可能为 None（单测直调），兜底成默认随机条数。
        n = 0
        if request is not None:
            try:
                n = int(request.query_params.get("n", "0"))
            except (ValueError, TypeError):
                n = 0

        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]
        if not candidates:
            return PlainTextResponse("")

        picks = _sample_dream_material(candidates, n)
        if not picks:
            return PlainTextResponse("")

        parts = []
        for b in picks:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{redact_text(strip_wikilinks(b['content'][:200]))}"
            )

        text = "[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts)
        return PlainTextResponse(redact_text(_append_anchor_index(text, _format_anchor_index(picks))))
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# LMC-5 hook bridges: exact SessionEnd ingest + per-turn recall
# LMC-5 挂钩桥：SessionEnd 原始事件入账 + 每轮召回
# =============================================================
_LMC5_HOOK_MAX_BODY_BYTES = 32 * 1024 * 1024
_LMC5_HOOK_MAX_EVENTS = 5000
_LMC5_RECALL_MAX_BODY_BYTES = 64 * 1024
_LMC5_RECALL_MAX_PROMPT_CHARS = 20000


async def _read_bounded_json_object(request, *, max_bytes: int) -> dict:
    raw = await request.body()
    if len(raw) > max_bytes:
        raise ValueError("request body too large")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("JSON object required")
    return body


@mcp.custom_route("/lmc5/raw-events", methods=["POST"])
async def lmc5_raw_events_hook(request):
    """Atomically append one SessionEnd transcript batch.

    ``payload`` is the exact JSONL source line, not a model summary. The
    ledger owns deduplication by ``(session_id, source_event_id)`` and rolls
    the whole batch back on any identity conflict.
    """
    from starlette.responses import JSONResponse

    auth_state = _hook_auth_state(request)
    if auth_state == "unconfigured":
        return JSONResponse(
            {"error": "hook authentication is not configured"},
            status_code=503,
        )
    if auth_state != "authorized":
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        body = await _read_bounded_json_object(
            request,
            max_bytes=_LMC5_HOOK_MAX_BODY_BYTES,
        )
        if set(body) != {"schema_version", "session_id", "events"}:
            raise ValueError("raw-event contract fields do not match")
        if body["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        session_id = body["session_id"]
        events = body["events"]
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id required")
        if not isinstance(events, list) or not events:
            raise ValueError("non-empty events list required")
        if len(events) > _LMC5_HOOK_MAX_EVENTS:
            raise ValueError("too many events")

        normalized = []
        for event in events:
            if not isinstance(event, dict) or set(event) != {
                "source_event_id",
                "payload",
            }:
                raise ValueError("event contract fields do not match")
            source_event_id = event["source_event_id"]
            payload = event["payload"]
            if not isinstance(source_event_id, str) or not source_event_id.strip():
                raise ValueError("source_event_id required")
            if not isinstance(payload, str) or not payload.strip():
                raise ValueError("exact JSONL payload required")
            # Retain the exact line, but require every event to be an
            # independently valid JSON object before the atomic ledger call.
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("transcript event must be a JSON object")
            normalized.append(
                {
                    "session_id": session_id,
                    "source_event_id": source_event_id,
                    "payload": payload,
                }
            )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        # One SQLite transaction with bounded input.  Keep it in this request
        # rather than acknowledging before a detached worker has fsynced it.
        # Ledger setup and FULL-synchronous SQLite writes can wait on a busy
        # writer.  Keep the HTTP event loop responsive, but await the worker
        # before issuing the durable acknowledgement.
        def append_under_ingest_guard():
            # Acquire before constructing the lazy ledger and retain the lease
            # until its FULL-synchronous append transaction has completed.
            with shared_ingest_guard():
                return _get_lmc5_ledger().append_raw_events(normalized)

        results = await _await_daemon_thread(append_under_ingest_guard)
    except RawIngestBusy:
        return JSONResponse(
            {
                "error": "raw-event ingest is paused",
                "code": "raw_ingest.busy",
            },
            status_code=503,
        )
    except RawIngestGuardError:
        logger.exception("LMC-5 raw-ingest guard is unavailable")
        return JSONResponse(
            {
                "error": "raw-event ingest is unavailable",
                "code": "raw_ingest.unavailable",
            },
            status_code=503,
        )
    except LedgerConflictError:
        return JSONResponse({"error": "raw event identity conflict"}, status_code=409)
    except LedgerValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except LedgerCorruptionError:
        logger.exception("LMC-5 raw-event ledger is corrupt")
        return JSONResponse({"error": "raw-event ledger unavailable"}, status_code=503)
    except LedgerError:
        logger.exception("LMC-5 raw-event ingest failed")
        return JSONResponse({"error": "raw-event ingest failed"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "acknowledged": len(results),
            "inserted": sum(1 for result in results if result.created),
        }
    )


async def _read_lmc5_night_request(
    request,
    *,
    max_bytes: int = 1024,
) -> bytes:
    """Read the tiny scheduler contract without accepting an unbounded body."""
    stream = getattr(request, "stream", None)
    if callable(stream):
        chunks: list[bytes] = []
        total = 0
        async for chunk in stream():
            if not isinstance(chunk, bytes):
                raise ValueError("night request body is not bytes")
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("night request body is too large")
            chunks.append(chunk)
        return b"".join(chunks)
    raw = await request.body()
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise ValueError("night request body is invalid")
    return raw


@mcp.custom_route("/api/maintenance/lmc5-night", methods=["POST"])
async def lmc5_night_maintenance(request):
    """Run one authenticated, single-flight conservative LMC-5 night job."""
    from starlette.responses import JSONResponse

    if not _lmc5_night_enabled():
        return JSONResponse(
            {"ok": False, "code": "night.disabled"},
            status_code=503,
        )
    try:
        raw = await _read_lmc5_night_request(request)
        if raw.strip():
            body = json.loads(raw)
            if body != {"schema_version": 1}:
                raise ValueError("night request contract fields do not match")
    except (
        UnicodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        ValueError,
        TypeError,
    ):
        return JSONResponse(
            {"ok": False, "code": "request.invalid"},
            status_code=400,
        )

    try:
        result = await _get_lmc5_night_runtime().run_once()
    except NightRunRuntimeError as exc:
        status_code = 409 if exc.code in {"run.busy", "run.raced"} else 503
        logger.warning("LMC-5 night runtime stopped: %s", exc.code)
        return JSONResponse(
            {"ok": False, "code": exc.code},
            status_code=status_code,
        )
    except NightRunCoordinatorError as exc:
        logger.error("LMC-5 night run failed closed: %s", exc.code)
        return JSONResponse(
            {"ok": False, "code": exc.code},
            status_code=503,
        )
    except (LedgerError, OSError, ValueError):
        logger.error("LMC-5 night runtime unavailable")
        return JSONResponse(
            {"ok": False, "code": "night.unavailable"},
            status_code=503,
        )
    except Exception:
        logger.error("LMC-5 night runtime stopped unexpectedly")
        return JSONResponse(
            {"ok": False, "code": "night.unavailable"},
            status_code=503,
        )

    complete = result.stage == "complete"
    degraded = result.stage == "deferred"
    logger.info(
        "LMC-5 conservative night run finished: run_id=%s stage=%s "
        "complete=%s degraded=%s already_complete=%s counts=%s",
        result.run_id,
        result.stage,
        complete,
        degraded,
        result.already_complete,
        dict(result.counts),
    )
    return JSONResponse(
        {
            "ok": True,
            "contract": "lmc5-conservative-stage1",
            "run_id": result.run_id,
            "local_date": result.local_date,
            "stage": result.stage,
            "already_complete": result.already_complete,
            "complete": complete,
            "degraded": degraded,
            "cutoff_utc": result.cutoff_utc,
            "snapshot_manifest_sha256": result.snapshot_manifest_sha256,
            "counts": result.counts,
            "deferred_axes": ["Y", "Z", "E"],
        }
    )


@mcp.custom_route("/lmc5/recall-hook", methods=["POST"])
async def lmc5_recall_hook(request):
    """Run the normal authoritative recall pipeline for one user prompt."""
    from starlette.responses import JSONResponse

    auth_state = _hook_auth_state(request)
    if auth_state == "unconfigured":
        return JSONResponse(
            {"error": "hook authentication is not configured"},
            status_code=503,
        )
    if auth_state != "authorized":
        return JSONResponse({"error": "forbidden"}, status_code=403)

    try:
        body = await _read_bounded_json_object(
            request,
            max_bytes=_LMC5_RECALL_MAX_BODY_BYTES,
        )
        allowed_fields = {"schema_version", "prompt", "session_id"}
        if set(body) - allowed_fields or not {"schema_version", "prompt"} <= set(body):
            raise ValueError("recall-hook contract fields do not match")
        if body["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        prompt = body["prompt"]
        session_id = body.get("session_id", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt required")
        if len(prompt) > _LMC5_RECALL_MAX_PROMPT_CHARS:
            raise ValueError("prompt too large")
        if not isinstance(session_id, str) or len(session_id) > 512:
            raise ValueError("invalid session_id")
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    strict_token = _strict_recall_errors.set(True)
    try:
        result = await breath(
            query=prompt.strip(),
            max_tokens=min(BREATH_DEFAULT_MAX_TOKENS, 6000),
            max_results=BREATH_DEFAULT_MAX_RESULTS,
            relation_depth=2,
            session_id=session_id.strip(),
            include_images=False,
            include_body_state=False,
            reset_body_state=False,
        )
    except Exception:
        logger.exception("LMC-5 per-turn recall failed")
        return JSONResponse({"error": "recall failed"}, status_code=500)
    finally:
        _strict_recall_errors.reset(strict_token)

    if isinstance(result, str):
        if result in {
            "检索过程出错，请稍后重试。",
            "记忆系统暂时无法访问。",
            "读取 feel 失败。",
        }:
            logger.error("LMC-5 recall returned an operational failure sentinel")
            return JSONResponse({"error": "recall failed"}, status_code=500)
        context = result
    elif isinstance(result, list):
        context = "\n".join(
            value
            for item in result
            if isinstance((value := getattr(item, "text", None)), str)
        )
    else:
        context = str(result)
    return JSONResponse({"ok": True, "context": context})


# =============================================================
# Internal helper: resolve world filter for breath
# 内部辅助：根据 world 显式参数 + 全局 current_world 决定过滤集合
# 返回 None 表示不过滤（"all" 模式），否则返回 filter list。
# =============================================================
def _resolve_world_filter(world_param: str, current_world: str):
    wp = (world_param or "").strip()
    if wp.lower() == "all":
        return None
    if wp:
        return [x.strip() for x in wp.split(",") if x.strip()]
    return [(current_world or "").strip()]


def _metadata_list(value) -> list:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if value is None:
        return []
    return [str(value)]


def _entity_retrieval_keys(entities: list[dict] | None) -> list[str]:
    """Extract exact, already-validated entity mentions for bucket clothing."""
    return [
        str(entity.get("mention") or "").strip()
        for entity in (entities or [])
        if isinstance(entity, dict) and str(entity.get("mention") or "").strip()
    ]


def _bucket_primary_domain_matches(meta: dict, domain: list) -> bool:
    if not domain:
        return True
    primary = str(domain[0])
    return primary in _metadata_list(meta.get("domain", []))


def _has_redactable_secret(value: str) -> bool:
    if not value:
        return False
    text = str(value)
    return redact_embedding_input(text) != text


def _is_merge_protected_bucket(bucket: dict, incoming_domain: list = None, incoming_chord_tag: str = "") -> bool:
    meta = bucket.get("metadata", {}) or {}
    if meta.get("pinned") or meta.get("protected"):
        return True
    if meta.get("type") in ("permanent", "feel"):
        return True

    if _has_redactable_secret(bucket.get("content", "")):
        return True

    bucket_domains = set(_metadata_list(meta.get("domain", [])))
    incoming_domains = set(_metadata_list(incoming_domain or []))
    if (bucket_domains | incoming_domains) & PROTECTED_RESOLVE_DOMAINS:
        return True

    # Chord-tagged memories carry affective color. Do not auto-rewrite them via dedupe merge.
    if meta.get("chord_tag") or (incoming_chord_tag and incoming_chord_tag.strip()):
        return True
    return False


def _passes_merge_candidate_filters(
    bucket: dict,
    domain: list,
    world_filter: list,
    incoming_chord_tag: str = "",
) -> bool:
    meta = bucket.get("metadata", {}) or {}
    if _is_merge_protected_bucket(bucket, domain, incoming_chord_tag):
        return False
    if world_filter is not None:
        wf_set = {str(w).strip() for w in world_filter}
        if not world_matches(meta.get("world", ""), wf_set):
            return False
    return _bucket_primary_domain_matches(meta, domain)


def _merge_candidate_passes_threshold(bucket: dict) -> bool:
    merge_cfg = config.get("merge", {}) or {}
    keyword_threshold = float(merge_cfg.get("keyword_threshold", config.get("merge_threshold", 85)))
    vector_threshold = float(merge_cfg.get("vector_threshold", 0.78))
    kw_score = float(bucket.get("merge_keyword_score", bucket.get("score", 0)) or 0)
    vec_sim = float(bucket.get("merge_vector_similarity", 0) or 0)
    return kw_score > keyword_threshold or vec_sim >= vector_threshold


async def _find_merge_candidates(
    content: str,
    domain: list,
    world_filter: list,
    incoming_chord_tag: str = "",
) -> list[dict]:
    merge_cfg = config.get("merge", {}) or {}
    keyword_limit = int(merge_cfg.get("keyword_limit", 5))
    vector_limit = int(merge_cfg.get("vector_limit", 8))
    candidate_limit = int(merge_cfg.get("candidate_limit", max(keyword_limit, vector_limit)))
    vector_floor = float(merge_cfg.get("vector_floor", 0.50))

    try:
        keyword_matches = await bucket_mgr.search(
            content,
            limit=keyword_limit,
            domain_filter=domain or None,
            world_filter=world_filter,
        )
    except Exception as e:
        logger.warning(f"Keyword merge search failed / 关键词合并搜索失败: {e}")
        keyword_matches = []

    keyword_matches = [
        b for b in keyword_matches
        if _passes_merge_candidate_filters(b, domain, world_filter, incoming_chord_tag)
    ]

    try:
        vector_raw = await embedding_engine.search_similar(content, top_k=vector_limit)
        vector_ranked = [(bid, sim) for bid, sim in vector_raw if sim >= vector_floor]
    except Exception as e:
        logger.warning(f"Vector merge search failed, using keyword only / 向量合并搜索失败: {e}")
        vector_ranked = []

    keyword_by_id = {b["id"]: b for b in keyword_matches}
    keyword_scores = {b["id"]: float(b.get("score", 0) or 0) for b in keyword_matches}
    vector_scores = {bid: float(sim) for bid, sim in vector_ranked}
    keyword_ranked = [(bid, score) for bid, score in keyword_scores.items()]

    rrf_cfg = config.get("rrf", {}) or {}
    fused_pairs = rrf_fuse(
        keyword_ranked,
        vector_ranked,
        k=rrf_cfg.get("k", 60),
        keyword_weight=rrf_cfg.get("keyword_weight", 1.0),
        vector_weight=rrf_cfg.get("vector_weight", 1.0),
    )

    candidates = []
    for bid, fused_score in fused_pairs:
        if len(candidates) >= candidate_limit:
            break
        if bid in keyword_by_id:
            bucket = keyword_by_id[bid]
        else:
            bucket = await bucket_mgr.get(bid)
            if not bucket:
                continue
        if not _passes_merge_candidate_filters(bucket, domain, world_filter, incoming_chord_tag):
            continue

        kw_score = keyword_scores.get(bid, 0.0)
        vec_sim = vector_scores.get(bid, 0.0)
        bucket["merge_keyword_score"] = round(kw_score, 2)
        bucket["merge_vector_similarity"] = round(vec_sim, 4)
        bucket["merge_fused_score"] = round(fused_score, 6)
        bucket["score"] = round(max(kw_score, vec_sim * 100.0, fused_score * 1000.0), 2)
        if bid not in keyword_by_id:
            bucket["vector_match"] = True
        candidates.append(bucket)
    return candidates


_ARBITRATION_CONTEXT_BLOCK_RE = re.compile(r"\n?\[ARBITRATION_CONTEXT\].*?\[/ARBITRATION_CONTEXT\]\n?", re.DOTALL)


def _strip_arbitration_context(content: str) -> str:
    if not content:
        return content
    return _ARBITRATION_CONTEXT_BLOCK_RE.sub("\n", content).strip()


async def _apply_bucket_update(
    bucket_id: str,
    updates: dict,
    *,
    entities: list[dict] | None = None,
    actor: str = "system",
    expected_content_hash: str = "",
    expected_revision_hash: str = "",
) -> bool:
    """Apply a bucket update and keep derived indexes current.

    Markdown remains the source of truth.  Embedding/entity refresh failures
    are best-effort sidecar failures and must never turn an already committed
    update into a second newly-created bucket.
    """
    committed = False
    try:
        async with bucket_mgr._maintenance_barrier.shared_async():
            try:
                success = await bucket_mgr.update(
                    bucket_id,
                    actor=actor,
                    expected_content_hash=expected_content_hash,
                    expected_revision_hash=expected_revision_hash,
                    **updates,
                )
            except Exception as exc:
                success = False
                logger.warning(
                    "Bucket update raised before status was known: %s: %s",
                    bucket_id,
                    type(exc).__name__,
                )
            # BucketManager writes Markdown before committing its audit row.
            # If that later bookkeeping step fails it reports False even
            # though the source of truth already contains the new body.  Read
            # back once so grow cannot then create a duplicate bucket.
            if not success and "content" in updates:
                try:
                    latest = await bucket_mgr.get(bucket_id)
                except Exception:
                    latest = None
                latest_meta = (latest or {}).get("metadata", {}) or {}
                landed = bool(latest)
                for key, expected_value in updates.items():
                    actual_value = (
                        latest.get("content")
                        if key == "content"
                        else latest_meta.get(key)
                    )
                    if actual_value != expected_value:
                        landed = False
                        break
                if landed:
                    success = True
                    logger.warning(
                        "Bucket update reported failure after content landed: %s",
                        bucket_id,
                    )
            if not success:
                return False
            committed = True
            if "content" in updates:
                try:
                    await embedding_engine.generate_and_store(
                        bucket_id,
                        updates["content"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Embedding refresh after bucket update failed: %s: %s",
                        bucket_id,
                        type(exc).__name__,
                    )
    except Exception as exc:
        logger.warning(
            "Bucket update %s: %s: %s",
            "post-commit cleanup failed" if committed else "failed",
            bucket_id,
            type(exc).__name__,
        )
        if not committed:
            return False
    if "content" in updates:
        try:
            await _synchronize_bucket_entities(bucket_id, updates["content"], entities)
        except Exception as exc:
            logger.warning(
                "Entity refresh after bucket update failed: %s: %s",
                bucket_id,
                type(exc).__name__,
            )
    return True


async def _recall_before_write_decision(
    content: str,
    world: str,
    domain: list,
) -> str:
    """Run one read-only top-5 recall and return a strict three-state decision.

    Any retrieval/model/parse error deliberately becomes ``new``.  Incoming
    memory must never be dropped merely because the advisory model is down.
    """
    if _has_redactable_secret(content):
        logger.info("recall-before-write skipped secret-bearing content; fallback=new")
        return "new"
    capture: list[dict] = []
    capture_token = _breath_candidate_capture.set(capture)
    try:
        await breath(
            query=content,
            max_tokens=4000,
            max_results=5,
            domain=",".join(_metadata_list(domain)),
            world=world,
            relation_depth=0,
            session_id="",
            include_images=False,
            include_body_state=False,
        )
        trusted_candidates = [
            item
            for item in capture[:5]
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item["id"].strip()
        ]
        candidate_ids = list(dict.fromkeys(
            item["id"].strip() for item in trusted_candidates
        ))
        if not candidate_ids:
            logger.info("recall-before-write found no candidate; decision=new")
            return "new"
        recalled_text = "\n\n".join(
            f"<candidate bucket_id={json.dumps(item['id'], ensure_ascii=False)}>\n"
            f"{str(item.get('summary') or '')}\n</candidate>"
            for item in trusted_candidates
            if item["id"].strip() in candidate_ids
        )
        decision = await dehydrator.arbitrate_recall_before_write(
            content,
            recalled_text,
            candidate_ids,
        )
        if decision != "new":
            action, separator, bucket_id = str(decision).partition(":")
            if (
                separator != ":"
                or action not in {"merge", "supersede"}
                or bucket_id not in candidate_ids
            ):
                raise RuntimeError(
                    "recall-before-write adapter escaped candidate allowlist"
                )
            decision = f"{action}:{bucket_id}"
        logger.info(
            "recall-before-write decision=%s candidates=%d",
            decision.split(":", 1)[0],
            len(candidate_ids),
        )
        return decision
    except Exception as exc:
        logger.warning(
            "recall-before-write failed; fallback=new: %s",
            type(exc).__name__,
        )
        return "new"
    finally:
        _breath_candidate_capture.reset(capture_token)


async def _create_operational_status_successor(
    *,
    target: dict,
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str,
    world: str,
    chord_tag: str,
    detected_senses: list,
    entities: list[dict] | None,
) -> tuple[str, str, bool]:
    """Preserve both status snapshots and atomically mark their validity.

    Markdown remains the source text store.  If the additive sidecar fails,
    both buckets still exist and recall fails open to ``validity:unknown``.
    """
    async with bucket_mgr._maintenance_barrier.shared_async():
        new_bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name or None,
            retrieval_keys=_entity_retrieval_keys(entities),
            world=world,
            chord_tag=chord_tag,
            sense=detected_senses or None,
            actor="grow:recall-before-write:status-successor",
        )
        try:
            await embedding_engine.generate_and_store(new_bucket_id, content)
        except Exception as exc:
            logger.warning(
                "Embedding for operational status successor failed: %s: %s",
                new_bucket_id,
                type(exc).__name__,
            )

    new_meta: dict = {}
    try:
        new_bucket = await bucket_mgr.get(new_bucket_id)
        if not new_bucket:
            raise RuntimeError("new operational status bucket is not readable")
        target_meta = target.get("metadata", {}) or {}
        new_meta = new_bucket.get("metadata", {}) or {}
        marker = _get_operational_status_validity_store().mark_supersession(
            old_bucket_id=str(target.get("id") or target_meta.get("id") or ""),
            new_bucket_id=new_bucket_id,
            old_valid_at=str(
                target_meta.get("event_at")
                or target_meta.get("created")
                or target_meta.get("recorded_at")
                or now_iso()
            ),
            new_valid_at=str(
                new_meta.get("event_at")
                or new_meta.get("created")
                or new_meta.get("recorded_at")
                or now_iso()
            ),
            source_ref="grow:recall-before-write:supersede",
        )
        logger.info(
            "Operational status supersession recorded old=%s new=%s key=%s",
            target.get("id"),
            new_bucket_id,
            marker["status_key"],
        )
    except Exception as exc:
        logger.warning(
            "Operational status marker failed open after durable create: %s",
            type(exc).__name__,
        )
    await _synchronize_bucket_entities(new_bucket_id, content, entities)
    created_name = str((new_meta or {}).get("name") or "").strip()
    return new_bucket_id, (created_name or name or new_bucket_id), False


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    world: str = "",
    chord_tag: str = "",
    require_self_contained: bool = False,
    recall_before_write: bool = False,
    entities: list[dict] | None = None,
) -> tuple[str, str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id, display_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID, 显示名, 是否合并)。
    """
    # grow validates every incoming item before this helper is called.  Do not
    # audit it a second time here: a mapping-only resolution deliberately
    # changes the text and no longer carries the original antecedent context.
    # The flag below guards only the LLM-produced merge result.

    # 五感入口层 v1：从内容识别感官标签（嗅/味/触/听），合并与新建两路都带上。
    detected_senses = detect_senses(content)

    # grow-only Phase 2.5: breath top-5 + small-model arbitration.  Explicit
    # ``new`` and every arbitration failure skip the legacy heuristic merge
    # below and go straight to the existing create path.
    if recall_before_write:
        decision = await _recall_before_write_decision(content, world, domain)
        action, separator, target_id = decision.partition(":")
        if separator == ":" and action in {"merge", "supersede"}:
            try:
                target = await bucket_mgr.get(target_id)
                if not target:
                    raise RuntimeError("selected recall candidate no longer exists")
                target_meta = target.get("metadata", {}) or {}
                if _is_merge_protected_bucket(target, domain, chord_tag):
                    raise RuntimeError("selected recall candidate is protected")
                if (target_meta.get("world", "") or "").strip() != (world or "").strip():
                    raise RuntimeError("selected recall candidate is from another world")
                if not _bucket_primary_domain_matches(target_meta, domain):
                    raise RuntimeError("selected recall candidate is from another domain")
                if build_supersedes_audit(target, content):
                    raise RuntimeError(
                        "selected recall candidate requires explicit Z review"
                    )

                if (
                    action == "supersede"
                    and _operational_status_validity_enabled()
                    and is_operational_status_fact(
                        str(target.get("content") or ""),
                        target_meta.get("domain"),
                        bucket_type=str(target_meta.get("type") or "dynamic"),
                        pinned=bool(target_meta.get("pinned")),
                        protected=bool(target_meta.get("protected")),
                    )
                    and is_operational_status_fact(content, domain)
                ):
                    return await _create_operational_status_successor(
                        target=target,
                        content=content,
                        tags=tags,
                        importance=importance,
                        domain=domain,
                        valence=valence,
                        arousal=arousal,
                        name=name,
                        world=world,
                        chord_tag=chord_tag,
                        detected_senses=detected_senses,
                        entities=entities,
                    )

                replacement = content
                update_kwargs = {"content": replacement}
                if action == "merge":
                    replacement = await dehydrator.merge(target.get("content", ""), content)
                    replacement = _strip_arbitration_context(replacement) or replacement
                    if require_self_contained:
                        replacement = await dehydrator.ensure_self_contained(
                            replacement,
                            source_context=f"{target.get('content', '')}\n\n{content}",
                        )
                    old_v = target_meta.get("valence", 0.5)
                    old_a = target_meta.get("arousal", 0.3)
                    structured_senses = senses_from_sensory({"metadata": target_meta})
                    merged_senses = union_senses(
                        target_meta.get("sense"),
                        detected_senses,
                        structured_senses,
                    )
                    update_kwargs = {
                        "content": replacement,
                        "tags": list(dict.fromkeys(
                            _metadata_list(target_meta.get("tags", []))
                            + _metadata_list(tags)
                        )),
                        "importance": max(
                            target_meta.get("importance", 5),
                            importance,
                        ),
                        "domain": list(dict.fromkeys(
                            _metadata_list(target_meta.get("domain", []))
                            + _metadata_list(domain)
                        )),
                        "valence": round((old_v + valence) / 2, 2),
                        "arousal": round((old_a + arousal) / 2, 2),
                    }
                    if merged_senses:
                        update_kwargs["sense"] = merged_senses
                else:
                    update_kwargs.update({
                        "tags": list(dict.fromkeys(_metadata_list(tags))),
                        "importance": importance,
                        "domain": list(dict.fromkeys(_metadata_list(domain))),
                        "valence": valence,
                        "arousal": arousal,
                        "sense": detected_senses,
                    })
                    if name:
                        update_kwargs["name"] = name
                expected_revision_hash = bucket_revision_hash(
                    target.get("content", ""),
                    target_meta,
                )
                update_ok = await _apply_bucket_update(
                    target_id,
                    update_kwargs,
                    entities=entities if action == "supersede" else None,
                    actor=f"grow:recall-before-write:{action}",
                    expected_revision_hash=expected_revision_hash,
                )
                if not update_ok:
                    raise RuntimeError("selected recall candidate could not be updated")
                logger.info(
                    "recall-before-write applied action=%s target=%s",
                    action,
                    target_id,
                )
                display_name = target_meta.get("name", target_id)
                if action == "supersede" and name:
                    display_name = name
                return (
                    target_id,
                    display_name,
                    True,
                )
            except Exception as exc:
                logger.warning(
                    "recall-before-write action failed; fallback=new: %s",
                    type(exc).__name__,
                )
        existing = []

    # world="" 即日常桶，只在日常桶之间合并；通用桶单独按通用合并。
    # 合并候选必须在同一个 world 内（避免日常桶被角色记忆合并污染或反过来）。
    world_filter = [(world or "").strip()]
    if recall_before_write or _has_redactable_secret(content):
        existing = []
    else:
        try:
            existing = await _find_merge_candidates(
                content=content,
                domain=domain,
                world_filter=world_filter,
                incoming_chord_tag=chord_tag,
            )
        except Exception as e:
            logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
            existing = []

    bucket = next((b for b in existing if _merge_candidate_passes_threshold(b)), None)
    if bucket:
        # --- Never merge into pinned/protected/permanent buckets ---
        # --- 不合并到钉选/保护/固化桶（这些桶分数恒定 999，标签网常常很宽，
        # ---  允许吸入会让它们变成"吸尘器"把所有相关 hold 都揽进去）---
        bmeta = bucket["metadata"]
        if not _is_merge_protected_bucket(bucket, domain, chord_tag):
            try:
                audit_entries = build_supersedes_audit(bucket, content)
                if audit_entries:
                    # A factual flip must remain two independently reviewable
                    # buckets.  Do not merge content, write supersedes metadata,
                    # or enqueue a candidate implicitly; the explicit Z
                    # dry-run/apply flow owns all three lifecycle transitions.
                    logger.info(
                        "Z conflict kept separate pending explicit review / "
                        "Z轴冲突保留为独立事实待显式审查: %s",
                        bucket["id"],
                    )
                else:
                    merged = await dehydrator.merge(bucket["content"], content)
                    merged = _strip_arbitration_context(merged) or merged
                    if require_self_contained:
                        merged = await dehydrator.ensure_self_contained(
                            merged,
                            source_context=f"{bucket['content']}\n\n{content}",
                        )
                    old_v = bucket["metadata"].get("valence", 0.5)
                    old_a = bucket["metadata"].get("arousal", 0.3)
                    merged_valence = round((old_v + valence) / 2, 2)
                    merged_arousal = round((old_a + arousal) / 2, 2)
                    update_kwargs = dict(
                        content=merged,
                        tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                        importance=max(bucket["metadata"].get("importance", 5), importance),
                        domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                        valence=merged_valence,
                        arousal=merged_arousal,
                    )
                    # 感官标签并入（合并不丢已有 sense、补上新内容触到的感官）
                    # 结构化 sensory.spicy/touch 也映射成 sense 通道：闭环另一半——带 sensory.* 的桶
                    # 既能被读到点燃身体，也能被「味觉/触觉」类 query 上浮（普鲁斯特钩子，小卷 #1）。
                    structured_senses = senses_from_sensory({"metadata": bmeta})
                    merged_senses = union_senses(
                        bmeta.get("sense"),
                        detected_senses,
                        structured_senses,
                    )
                    if merged_senses:
                        update_kwargs["sense"] = merged_senses
                    async with bucket_mgr._maintenance_barrier.shared_async():
                        await bucket_mgr.update(bucket["id"], **update_kwargs)
                        # --- Update embedding after merge ---
                        # 扫盘 #10：embedding 失败不再静默——语义检索会悄悄陈旧，至少留 warning
                        try:
                            await embedding_engine.generate_and_store(
                                bucket["id"],
                                merged,
                            )
                        except Exception as e:
                            logger.warning(
                                "Embedding update after merge failed / "
                                f"合并后向量更新失败: {bucket['id']}: {e}"
                            )
                    await _synchronize_bucket_entities(bucket["id"], merged, entities)
                    return (
                        bucket["id"],
                        bucket["metadata"].get("name", bucket["id"]),
                        True,
                    )
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    async with bucket_mgr._maintenance_barrier.shared_async():
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name or None,
            retrieval_keys=_entity_retrieval_keys(entities),
            world=world,
            chord_tag=chord_tag,
            sense=detected_senses or None,
        )
        # --- Generate embedding for new bucket ---
        # 扫盘 #10：失败留痕，否则新桶语义检索召不回且无任何日志线索
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception as e:
            logger.warning(f"Embedding for new bucket failed / 新桶向量生成失败: {bucket_id}: {e}")
    await _synchronize_bucket_entities(bucket_id, content, entities)
    get_created_bucket = getattr(bucket_mgr, "get", None)
    created_bucket = (
        await get_created_bucket(bucket_id)
        if callable(get_created_bucket)
        else None
    )
    created_metadata = (created_bucket or {}).get("metadata", {}) or {}
    display = str(created_metadata.get("name") or name or bucket_id)
    return bucket_id, display, False


# =============================================================
# Background backfill: hydrate relations for legacy buckets without edges
# 后台 backfill：给老桶补建关系网
# Lazy-started on first hold/breath call. Idempotent — only touches buckets
# whose `relations` field is empty/missing, so it can run safely on every
# server restart without redoing work.
# =============================================================
_backfill_started = False
# 扫盘 #11：check-and-set 加锁。asyncio 单线程本不会交错，但 MCP 工具可能被
# 线程池/多 loop 调起，无锁时并发首调会起多个 backfill 重复烧 LLM。
_backfill_start_lock = threading.Lock()


async def _startup_backfill_loop() -> None:
    """Walk eligible buckets without relations and run _auto_infer_edges on each.
    Rate-limited to ~1 bucket per 2s so the LLM API isn't hammered."""
    global _backfill_started
    try:
        await asyncio.sleep(30)  # let the server fully come up first
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.warning(f"Backfill list_all failed / 列桶失败: {e}")
            return

        candidates = [
            b for b in all_buckets
            if not b["metadata"].get("pinned")
            and not b["metadata"].get("protected")
            and b["metadata"].get("type") not in ("feel", "permanent")
            and not b["metadata"].get("resolved", False)
            and not b["metadata"].get("relations")
        ]
        candidates.sort(key=lambda b: b["id"])

        if not candidates:
            logger.info("Backfill: no eligible buckets / 没有需 backfill 的桶")
            return

        logger.info(
            f"Backfill starting: {len(candidates)} eligible buckets / "
            f"开始 backfill {len(candidates)} 个桶"
        )

        for i, b in enumerate(candidates):
            try:
                proposals = await _auto_infer_edges(
                    source_id=b["id"],
                    content=b["content"],
                    world=b["metadata"].get("world", ""),
                )
                if i % 5 == 0:
                    logger.info(
                        f"Backfill {i + 1}/{len(candidates)} | "
                        f"{b['id'][:6]} +{len(proposals)}提议"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Backfill bucket {b['id']} failed: {e}")
            await asyncio.sleep(2)  # rate-limit LLM calls

        logger.info(
            f"Backfill complete: {len(candidates)} buckets / "
            f"backfill 完成 {len(candidates)} 桶"
        )
    finally:
        # Don't reset flag on success — we don't want repeat passes within same
        # process. A server restart will re-trigger via _maybe_start_backfill.
        pass


def _startup_relation_backfill_enabled() -> bool:
    """Return whether the expensive legacy relation sweep was explicitly enabled.

    A production library can contain thousands of multi-vector embeddings.  The
    legacy sweep performs a full similarity search for every eligible bucket, so
    starting it implicitly from a user-facing request can starve the HTTP event
    loop for minutes.  Keep the manual ``backfill_relations`` tool available,
    but require an explicit maintenance opt-in for the unbounded background pass.
    """
    maintenance = config.get("maintenance", {})
    return (
        isinstance(maintenance, dict)
        and maintenance.get("startup_relation_backfill") is True
    )


def _maybe_start_backfill() -> None:
    """Lazy start the legacy backfill only after an explicit maintenance opt-in."""
    global _backfill_started
    if not _startup_relation_backfill_enabled():
        return
    if _backfill_started:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop yet, will retry on next call
    with _backfill_start_lock:
        if _backfill_started:
            return
        _backfill_started = True
    loop.create_task(_startup_backfill_loop())
    logger.info("Backfill task scheduled (T-30s) / backfill 已排程")


# =============================================================
# Helper: auto-edge inference for newly created bucket
# 工具：新桶自动建边
# Wraps embedding-similar + keyword-search candidate gathering, LLM relation
# inference via dehydrator, then write safe edges or queue dangerous proposals.
# All failures are swallowed so this never blocks hold.
# 包装 embedding 邻居 + 关键词搜索拿候选，dehydrator LLM 判边：
# kin/explains 经审计幂等写入；因果/取代类只挂具名待审。
# 所有失败吞掉——绝不阻塞 hold 主流程。
# =============================================================
async def _auto_infer_edges(
    source_id: str, content: str, world: str = ""
) -> list[dict]:
    """Commit safe inferred edges and queue dangerous ones for named review."""
    if not content or not content.strip():
        return []

    def _relation_display_name(bucket: dict | None, bucket_id: str) -> str:
        metadata = (
            bucket.get("metadata", {})
            if isinstance(bucket, dict)
            else {}
        )
        raw_name = str(metadata.get("name") or "").strip()
        if raw_name and raw_name != bucket_id:
            return raw_name[:160]
        domains = metadata.get("domain", [])
        if isinstance(domains, str):
            domains = [domains]
        if isinstance(domains, list):
            topic = next(
                (str(value).strip() for value in domains if str(value).strip()),
                "",
            )
            if topic:
                return f"{topic} · {bucket_id}"[:160]
        return bucket_id

    source_bucket = await bucket_mgr.get(source_id)
    source_name = _relation_display_name(source_bucket, source_id)

    # --- Gather candidates: vector neighbors + keyword search, dedup, exclude self ---
    candidate_ids: list[str] = []
    seen = {source_id}

    try:
        vec_hits = await embedding_engine.search_similar(content, top_k=8)
        for bid, _score in vec_hits:
            if bid not in seen:
                candidate_ids.append(bid)
                seen.add(bid)
    except Exception as e:
        logger.warning(f"Vector candidate fetch failed / 向量候选失败: {e}")

    try:
        kw_hits = await bucket_mgr.search(content, limit=5)
        for b in kw_hits:
            bid = b.get("id")
            if bid and bid not in seen:
                candidate_ids.append(bid)
                seen.add(bid)
    except Exception as e:
        logger.warning(f"Keyword candidate fetch failed / 关键词候选失败: {e}")

    if not candidate_ids:
        return []

    # --- Build candidate list with dehydrated summaries (cap 8 to bound LLM cost) ---
    candidates: list[dict] = []
    wf_set = {(world or "").strip()}
    for bid in candidate_ids[:8]:
        b = await bucket_mgr.get(bid)
        if not b:
            continue
        # Skip cross-world candidates: don't link daily↔roleplay buckets
        # 跨世界候选跳过：避免日常↔角色扮演桶被自动连边
        b_world = b["metadata"].get("world", "")
        if not world_matches(b_world, wf_set):
            continue
        try:
            summary = await dehydrator.dehydrate(
                strip_wikilinks(b["content"]),
                {k: v for k, v in b["metadata"].items() if k != "tags"},
            )
        except Exception:
            summary = (b["content"] or "")[:200]
        candidates.append({
            "id": bid,
            "name": _relation_display_name(b, bid),
            "summary": summary,
        })

    if not candidates:
        return []

    edges = await dehydrator.infer_relations(content, candidates)
    if not edges:
        return []

    cand_name_by_id = {c["id"]: c["name"] for c in candidates}
    proposed: list[dict] = []
    queued = 0
    applied = 0
    for edge in edges:
        etype = str(edge.get("type") or "").strip()
        target = str(edge.get("target") or "").strip()
        if etype not in RELATION_TYPES or target not in cand_name_by_id:
            continue
        try:
            outcome = {
                "type": etype,
                "target": target,
                "target_name": cand_name_by_id.get(target, target),
                "note": edge.get("note", ""),
            }
            if etype in SAFE_RELATION_TYPES:
                if not await bucket_mgr.add_relation(
                    source_id,
                    target,
                    etype,
                    edge.get("note", ""),
                    actor="lmc5-safe-relation",
                ):
                    continue
                applied += 1
                outcome["status"] = "applied"
                proposed.append(outcome)
                continue
            if etype not in REVIEW_RELATION_TYPES:
                continue
            if not _review_gate("relation_review"):
                logger.info(
                    "Dangerous relation proposal skipped: review gate disabled "
                    "source=%s target=%s type=%s",
                    source_id,
                    target,
                    etype,
                )
                continue
            entry = make_relation_entry(
                source_id,
                target,
                etype,
                edge.get("note", ""),
                source_name=source_name,
                target_name=cand_name_by_id.get(target, target),
            )
            if _get_review_queue().enqueue(entry):
                queued += 1
            outcome["status"] = "pending_review"
            proposed.append(outcome)
        except Exception as e:
            logger.warning(
                "关系提议入队失败（不阻塞）/ relation proposal enqueue failed: %s",
                type(e).__name__,
            )
    if queued:
        logger.info(
            "Machine relation inference queued %d proposal(s); graph unchanged",
            queued,
        )
    if applied:
        logger.info(
            "Machine relation inference applied %d safe audited edge(s)",
            applied,
        )
    return proposed


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = BREATH_DEFAULT_MAX_TOKENS,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = BREATH_DEFAULT_MAX_RESULTS,
    world: str = "",
    relation_depth: int = 1,
    since: str = "",
    until: str = "",
    session_id: str = "",
    policy: str = "search",
    include_images: bool = True,
    include_body_state: bool = True,
    reset_body_state: bool = False,
) -> str | list[TextContent | ImageContent]:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认6000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制注入数量上限(默认8,最大50; 内部仍先召回20条给过滤器)。world=过滤世界:留空走全局current_world(日常时只出日常+通用、角色扮演时只出该世界+通用),"all"跳过过滤,"旧世界"/"当前世界"等显式指定。world="通用"的桶永远跟着出。relation_depth=沿安全关系边双向召回邻居的跳数(默认1,0=关闭,最大2)，关联证据单独列出且不改变主排序。since/until=按桶 created 时间范围过滤,接受 ISO 8601("2026-05-01"/"2026-05-01T12:00:00")、关键字("now"/"today"/"yesterday")、相对偏移("-7d"/"-3h"/"-30m"/"+1d"),浮现模式不过滤 pinned/protected。session_id=同一会话内对已浮现动态桶去重。include_images=True时,白名单图桶会随文本返回 MCP image content。include_body_state=False时只关闭外部身体状态块,不改变记忆检索。reset_body_state=True时先清零 v0 外部身体状态,用于 A/B 盲测卫生。"""
    with recall_stage("setup"):
        await _ensure_decay_background()
        await _ensure_consolidation_background()
        await episode_engine.ensure_started()
        _maybe_start_backfill()
    max_results = max(1, min(max_results, 50))
    max_tokens = max(1000, min(max_tokens, 20000))
    recall_limit = max(BREATH_RECALL_POOL_SIZE, max_results)
    # Anchor 原生区分 conversation/reflex 与主动 search。Ombre 只复用这份
    # 策略合同，不照搬两边不可比较的绝对分数；未知值安全回到既有 search。
    requested_policy = str(policy or "search").strip().lower()
    recall_policy = _normalize_anchor_recall_policy(requested_policy)
    if recall_policy != requested_policy:
        logger.warning("Unknown recall policy %r; using search", requested_policy)
    allow_empty_recall = recall_policy in {"conversation", "reflex"}

    # --- Resolve world filter once (used by all modes) ---
    # --- 解析 world filter：显式参数 > current_world ---
    world_filter = _resolve_world_filter(world, config.get("current_world", ""))
    wf_set = {str(w).strip() for w in world_filter} if world_filter is not None else None

    # --- Resolve since/until once (shared by surfacing/feel/search modes) ---
    # --- 解析时间范围：无法解析的参数静默忽略，不报错 ---
    created_after = parse_relative_time(since) if since else None
    created_before = parse_relative_time(until) if until else None

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"
        all_buckets = [bucket for bucket in all_buckets if _is_main_recall_bucket(bucket)]

        # --- Pinned/protected buckets: always surface as core principles ---
        # --- 钉选桶：作为核心准则，始终浮现（不受 world 过滤影响）---
        pinned_buckets = [
            b for b in all_buckets
            if b["metadata"].get("pinned") or b["metadata"].get("protected")
        ]
        pinned_results = []
        for b in pinned_buckets:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), clean_meta, bucket=b)
                pinned_results.append(f"📌 [核心准则] [bucket_id:{b['id']}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 钉选桶脱水失败: {e}")
                continue

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not b["metadata"].get("pinned", False)
            and not b["metadata"].get("protected", False)
        ]

        # --- World filter on surfacing pool ---
        # --- 浮现池按 world 过滤：日常/角色扮演时不串场 ---
        if wf_set is not None:
            unresolved = [
                b for b in unresolved
                if world_matches(b["metadata"].get("world", ""), wf_set)
            ]

        # --- Time range filter on surfacing pool ---
        # --- 时间范围过滤：pinned/protected 不受影响（始终浮现），只过滤 unresolved 池 ---
        if created_after is not None or created_before is not None:
            from bucket_manager import _bucket_in_time_range
            unresolved = [
                b for b in unresolved
                if _bucket_in_time_range(b, created_after, created_before)
            ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} pinned, {len(unresolved)} unresolved "
            f"(world_filter={wf_set if wf_set is not None else 'all'})"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )
        scored = _filter_session_seen(scored, session_id)[:recall_limit]
        scored = await _ds_filter_candidates(
            "",
            scored,
            mode="surfacing",
            max_results=max_results,
            force_keep_ids=_exact_retrieval_key_ids("", scored),
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        for r in pinned_results:
            token_budget -= count_tokens_approx(r)

        # --- Feel buckets: emotional sediment, surface right after pinned ---
        # --- feel 桶:情感沉淀,紧跟核心准则浮现(独立池,不衰减)---
        feel_seen = {b["id"] for b in pinned_buckets}
        feel_results = []
        feel_buckets = []
        for b in _surface_feel_pool(all_buckets, feel_seen):
            try:
                fclean = {k: v for k, v in b["metadata"].items() if k != "tags"}
                fsummary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), fclean, bucket=b)
                feel_results.append(f"💧 [情感沉淀] [bucket_id:{b['id']}] {fsummary}")
                feel_buckets.append(b)
                token_budget -= count_tokens_approx(fsummary)
            except Exception as e:
                logger.warning(f"Failed to dehydrate feel bucket / 情感沉淀脱水失败: {e}")
                continue

        candidates = list(scored)
        if len(candidates) > 1:
            # Ensure highest-score bucket is first, shuffle rest from top-20
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        dynamic_buckets = []
        dynamic_ids = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), clean_meta, bucket=b)
                summary_tokens = count_tokens_approx(summary)
                if summary_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                score = decay_engine.calculate_score(b["metadata"])
                dynamic_results.append(f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}")
                dynamic_buckets.append(b)
                dynamic_ids.append(b["id"])
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        if not pinned_results and not dynamic_results and not feel_results:
            return _append_body_state_block(
                "权重池平静，没有需要处理的记忆。",
                [],
                session_id,
                include_body_state,
                reset_body_state,
            )

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if feel_results:
            parts.append("=== 情感沉淀 (feel) ===\n" + "\n---\n".join(feel_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        image_buckets = pinned_buckets + feel_buckets + dynamic_buckets
        text = "\n\n".join(parts)
        text = _append_body_state_block(
            text,
            image_buckets,
            session_id,
            include_body_state,
            reset_body_state,
        )
        _remember_session_seen_ids(session_id, dynamic_ids)
        return await _tool_result_with_optional_images(text, image_buckets, include_images)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [
                b
                for b in all_buckets
                if _is_main_recall_bucket(b) and b["metadata"].get("type") == "feel"
            ]
            if created_after is not None or created_before is not None:
                from bucket_manager import _bucket_in_time_range
                feels = [f for f in feels if _bucket_in_time_range(f, created_after, created_before)]
            feels.sort(
                key=lambda b: event_at_from_metadata(b["metadata"]) or "",
                reverse=True,
            )
            if not feels:
                return _append_body_state_block(
                    "没有留下过 feel。",
                    [],
                    session_id,
                    include_body_state,
                    reset_body_state,
                )
            results = []
            shown_feels = []
            for f in feels:
                created = event_at_from_metadata(f["metadata"]) or ""
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                shown_feels.append(f)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            text = "=== 你留下的 feel ===\n" + "\n---\n".join(results)
            text = _append_body_state_block(
                text,
                shown_feels,
                session_id,
                include_body_state,
                reset_body_state,
            )
            return await _tool_result_with_optional_images(text, shown_feels, include_images)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (RRF fusion of keyword + vector) ---
    # --- 有参数：检索模式（关键词 + 向量 RRF 融合）---
    start_recall_stage("query_prep")
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None
    e_recall_cfg = None
    e_query_emotion = None
    e_rows_by_bucket = {}
    try:
        e_recall_cfg = load_e_axis_recall_config(config)
        if e_recall_cfg.enabled:
            e_query_emotion = infer_query_emotion(
                query,
                valence_01=q_valence,
                arousal=q_arousal,
            )
            e_rows_by_bucket = await _e_axis_rows_cached(e_recall_cfg)
    except Exception as exc:
        # E is an optional behavioural projection.  A corrupt ledger or bad
        # live config fails closed to the relevance-first legacy path instead
        # of taking factual recall down with it.
        logger.warning(
            "E live projection unavailable; factual recall unchanged: %s",
            type(exc).__name__,
        )
        e_recall_cfg = None
        e_query_emotion = None
        e_rows_by_bucket = {}
    recall_query, raw_entity_ranked = _resolve_entity_recall(query)
    state_profile = _state_recall_profile(recall_query)
    state_seed_by_id: dict[str, dict] = {}
    intent_policy = _resolve_recall_policy(
        recall_query,
        base_recall_limit=recall_limit,
        requested_relation_depth=relation_depth,
    )
    if intent_policy["intent"] != "default":
        logger.info(
            "Intent-aware recall: "
            f"intent={intent_policy['intent']} "
            f"confidence={intent_policy['confidence']} "
            f"keyword_top_k={intent_policy['keyword_top_k']} "
            f"vector_top_k={intent_policy['vector_top_k']}"
        )
    finish_recall_stage("query_prep")

    query_angles = [recall_query]
    qe_cfg = config.get("query_expansion", {}) or {}
    qe_allowed = set(qe_cfg.get("allowed_intents") or ["recall", "relation", "temporal"])
    with recall_stage("expansion"):
        if qe_cfg.get("enabled", False) and intent_policy.get("intent") in qe_allowed:
            try:
                query_angles = await expand_query(
                    recall_query,
                    getattr(dehydrator, "client", None),
                    getattr(dehydrator, "model", "deepseek-chat"),
                    qe_cfg,
                ) or [recall_query]
            except Exception as e:
                logger.warning(f"Query expansion failed, using original / 查询扩展失败，回退原词: {e}")
                query_angles = [recall_query]
    if query_angles[0] != recall_query:
        query_angles = [recall_query] + [q for q in query_angles if q != recall_query]
    record_recall_metric("query_angle_count", len(query_angles))

    # Keyword channel (already filtered by world/domain/threshold inside)
    keyword_by_id: dict[str, dict] = {}
    try:
        with recall_stage("keyword_bucket_load"):
            keyword_candidates = await bucket_mgr.list_all(include_archive=False)
        record_recall_metric("keyword_bucket_count", len(keyword_candidates))
        with recall_stage("keyword_search"):
            for angle in query_angles:
                for bucket in await bucket_mgr.search(
                    angle,
                    limit=intent_policy["keyword_top_k"],
                    domain_filter=domain_filter,
                    world_filter=world_filter,
                    query_valence=q_valence,
                    query_arousal=q_arousal,
                    created_after=created_after,
                    created_before=created_before,
                    relevance_first=True,
                    # Keep a broad relevance-ranked keyword pool for RRF. The
                    # original-query literal/vector evidence gate below decides
                    # eligibility after both channels are available.
                    relevance_candidate_floor=0.0,
                    preloaded_buckets=keyword_candidates,
                ):
                    existing = keyword_by_id.get(bucket["id"])
                    if existing is None or bucket.get("score", 0) > existing.get("score", 0):
                        keyword_by_id[bucket["id"]] = bucket
        state_seed_by_id.update({
            str(bucket["id"]): bucket
            for bucket in keyword_by_id.values()
            if bucket.get("id") and _is_main_recall_bucket(bucket)
        })
        keyword_matches = _filter_z_fact_candidates(
            (
                bucket
                for bucket in keyword_by_id.values()
                if _is_main_recall_bucket(bucket)
            ),
            query=recall_query,
            intent=intent_policy["intent"],
        )
    except Exception as e:
        logger.error(
            "Keyword search failed / 关键词检索失败: %s",
            e,
            exc_info=True,
        )
        if _strict_recall_errors.get():
            raise RecallOperationalError("keyword_search_failed") from e
        return "检索过程出错，请稍后重试。"

    # Vector channel — sim>0.5 floor blocks high-cosine noise
    vector_scores: dict[str, float] = {}
    original_vector_scores: dict[str, float] = {}
    try:
        for angle_index, angle in enumerate(query_angles):
            for bid, sim in await embedding_engine.search_similar(
                angle,
                top_k=intent_policy["vector_top_k"],
            ):
                if sim <= 0.5:
                    continue
                if sim > vector_scores.get(bid, 0.0):
                    vector_scores[bid] = sim
                if angle_index == 0 and sim > original_vector_scores.get(bid, 0.0):
                    original_vector_scores[bid] = sim
        vector_ranked = list(vector_scores.items())
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")
        if _strict_recall_errors.get():
            raise RecallOperationalError("vector_search_failed") from e
        vector_ranked = []

    candidate_started_at = time.perf_counter()

    # Entity channel — linked ids are advisory until their content hash and the
    # same authority filters as vector-only candidates have been revalidated.
    entity_top_k, entity_weight = _entity_recall_settings()
    entity_store = _get_entity_store(initialize=False)
    entity_bucket_cache: dict[str, dict] = {}
    entity_ranked: list[tuple[str, float]] = []
    for bid, score in raw_entity_ranked[:entity_top_k]:
        try:
            bucket = keyword_by_id.get(bid) or await bucket_mgr.get(bid)
            if not bucket or not _passes_nonkeyword_recall_filters(
                bucket,
                world_filter_set=wf_set,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
            ):
                continue
            if entity_store is None or not entity_store.link_is_current(
                bid, bucket.get("content", "")
            ):
                continue
            state_seed_by_id[str(bid)] = bucket
            if not _filter_z_fact_candidates(
                [bucket],
                query=recall_query,
                intent=intent_policy["intent"],
            ):
                continue
            entity_bucket_cache[bid] = bucket
            entity_ranked.append((bid, score))
        except Exception as exc:
            logger.warning(
                "Entity candidate skipped after validation: %s",
                type(exc).__name__,
            )

    # RRF fusion of keyword + vector + the optional entity channel.
    rrf_cfg = config.get("rrf", {})
    keyword_scores = {
        b["id"]: float(b.get("score", 0) or 0)
        for b in keyword_matches
    }
    keyword_ranked = list(keyword_scores.items())
    channels = [
        (keyword_ranked, intent_policy["keyword_weight"]),
        (vector_ranked, intent_policy["vector_weight"]),
    ]
    if entity_ranked and entity_weight > 0:
        channels.append((entity_ranked, entity_weight))
    fused_pairs = lmc5_fuse_ranked_channels(
        channels,
        k=rrf_cfg.get("k", 60),
    )

    # Materialize fused list: reuse channel buckets, fetch vector-only ones.
    bucket_cache = {b["id"]: b for b in keyword_matches}
    bucket_cache.update(entity_bucket_cache)
    literal_candidate_floor = max(
        0.0,
        float(
            getattr(
                bucket_mgr,
                "literal_candidate_floor",
                (config.get("matching", {}) or {}).get(
                    "literal_candidate_floor",
                    40.0,
                ),
            )
        ),
    )
    topic_score_calculator = getattr(bucket_mgr, "_calc_topic_score", None)
    matches = []
    for bid, fused_score in fused_pairs:
        if bid in bucket_cache:
            b = bucket_cache[bid]
        else:
            # Vector-only bucket — fetch and re-apply filters that bucket_mgr.search applied
            b = await bucket_mgr.get(bid)
            if not b or not _passes_nonkeyword_recall_filters(
                b,
                world_filter_set=wf_set,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
            ):
                continue
            b["vector_match"] = True
        b.pop("entity_match", None)
        if bid in entity_bucket_cache:
            b["entity_match"] = True
        state_seed_by_id[str(bid)] = b
        if not _filter_z_fact_candidates(
            [b],
            query=recall_query,
            intent=intent_policy["intent"],
        ):
            continue
        fused_relevance = round(fused_score * 1000, 6)
        b["score"] = round(fused_relevance, 2)
        b["_fused_relevance_score"] = fused_relevance
        # Evaluate literal relevance against the audited entity-canonical query,
        # not a generated expansion angle.  For ordinary queries recall_query is
        # the original text; an explicit alias seed may safely replace only the
        # matched entity name.
        b["_literal_relevance_score"] = (
            round(topic_score_calculator(recall_query, b) * 100.0, 4)
            if callable(topic_score_calculator)
            else literal_candidate_floor
        )
        b["_vector_relevance_score"] = round(
            float(vector_scores.get(bid, 0.0)),
            6,
        )
        b["_original_vector_relevance_score"] = round(
            float(original_vector_scores.get(bid, 0.0)),
            6,
        )
        matches.append(b)

    # Generated expansion angles may improve ranking, but cannot introduce a
    # candidate that neither the original words nor original embedding support.
    matches = retain_original_query_supported_candidates(
        matches,
        literal_score=lambda bucket: max(
            float(bucket.get("_literal_relevance_score", 0) or 0),
            literal_candidate_floor if bucket.get("entity_match") else 0.0,
        ),
        original_vector_score=lambda bucket: bucket.get(
            "_original_vector_relevance_score",
            0,
        ),
        literal_floor=literal_candidate_floor,
    )[:recall_limit]

    # Relevance is the first ordering key.  Forgetting curve, sense and intent
    # remain useful, but may only adjust candidates inside one narrow fused
    # relevance band.
    sense_cfg = config.get("sense", {})
    sense_boost = (
        float(sense_cfg.get("recall_boost", 1.25))
        if sense_cfg.get("enabled", True) else 1.0
    )
    query_senses = set(detect_senses(query)) if sense_boost != 1.0 else set()
    for b in matches:
        tie_break_score = decay_engine.apply_retrieval_decay(
            b["_fused_relevance_score"],
            b["metadata"],
        )
        if query_senses:
            b_sense = b["metadata"].get("sense")
            if isinstance(b_sense, str):
                b_sense = [b_sense]
            if b_sense and set(b_sense) & query_senses:
                tie_break_score *= sense_boost
        intent_multiplier = bucket_intent_score_multiplier(b, intent_policy)
        if intent_multiplier != 1.0:
            tie_break_score *= intent_multiplier
        if e_recall_cfg is not None and e_query_emotion is not None:
            try:
                e_annotation = select_current_annotation(
                    e_rows_by_bucket.get(str(b.get("id") or ""), ()),
                    b,
                    e_recall_cfg,
                )
            except Exception as exc:
                logger.warning(
                    "E annotation rejected for recall candidate %s: %s",
                    b.get("id"),
                    type(exc).__name__,
                )
                e_annotation = None
            if e_annotation is not None:
                e_resonance = e_axis_resonance_score(
                    e_query_emotion,
                    e_annotation,
                )
                tie_break_score = apply_resonance_tie_break(
                    tie_break_score,
                    e_resonance,
                    weight=e_recall_cfg.tie_break_weight,
                )
                b["_e_axis_annotation"] = e_annotation
                b["_e_axis_resonance"] = e_resonance
        b["_non_relevance_tie_break_score"] = round(tie_break_score, 6)

    fused_band = max(
        0.0,
        float(
            (config.get("matching", {}) or {}).get(
                "fused_relevance_tie_band",
                0.35,
            )
        ),
    )
    matches = rank_within_relevance_bands(
        matches,
        relevance_score=lambda bucket: bucket.get(
            "_fused_relevance_score",
            0,
        ),
        tie_break_score=lambda bucket: bucket.get(
            "_non_relevance_tie_break_score",
            0,
        ),
        band_width=fused_band,
    )

    matches, content_suppressed, content_fingerprint_errors = _dedupe_recall_content(matches)
    if content_suppressed or content_fingerprint_errors:
        logger.info(
            "Recall content dedup: suppressed=%d fingerprint_errors=%d",
            content_suppressed,
            content_fingerprint_errors,
        )
    matches = _filter_session_seen(matches, session_id)
    record_recall_stage("candidate_processing", time.perf_counter() - candidate_started_at)
    with recall_stage("anchor_gate"):
        matches = _filter_anchor_policy_candidates(matches, recall_policy)
    candidate_started_at = time.perf_counter()
    matches = align_fact_state_candidates(
        matches,
        profile=state_profile,
        registry=_fact_slot_registry(),
    )
    state_link_budget = min(
        int(state_profile.get("state_link_limit", 0) or 0),
        max(0, max_results - (1 if matches else 0)),
    )
    state_link_candidates = await _state_link_recall_candidates(
        state_seed_by_id.values(),
        profile=state_profile,
        world_filter_set=wf_set,
        domain_filter=domain_filter,
        created_after=created_after,
        created_before=created_before,
        excluded_ids=(
            {str(bucket.get("id")) for bucket in matches if bucket.get("id")}
            | _session_seen_bucket_ids(list(state_seed_by_id.values()), session_id)
            | _load_session_seen_ids(session_id)
        ),
        limit=state_link_budget,
    )
    state_link_candidates = _filter_session_seen(state_link_candidates, session_id)
    record_recall_stage("candidate_processing", time.perf_counter() - candidate_started_at)
    with recall_stage("anchor_gate"):
        state_link_candidates = _filter_anchor_policy_candidates(
            state_link_candidates,
            recall_policy,
        )[:state_link_budget]
    with recall_stage("assembly"):
        set_recall_partial_result(_local_partial_recall_text(
            matches,
            max_results=max(0, max_results - len(state_link_candidates)),
            max_tokens=max_tokens,
            state_profile=state_profile,
        ))
    with recall_stage("ds_filter"):
        matches = await _ds_filter_candidates(
            recall_query,
            matches,
            mode="search",
            max_results=max(0, max_results - len(state_link_candidates)),
            force_keep_ids=_exact_retrieval_key_ids(recall_query, matches),
            allow_empty=allow_empty_recall,
        )

    # Reserve one existing result slot only when a retained primary seed has
    # a real same-thread neighbor. Queries without such a neighbor keep their
    # pre-X result budget unchanged.
    timeline_slot_reserved = False
    timeline_buckets = []
    timeline_fallback_matches = []
    if max_results > 1 and matches:
        try:
            # Reuse the full per-request snapshot already loaded for keyword
            # search; X must not add another 8k-bucket filesystem scan.
            timeline_buckets = list(keyword_candidates)
            primary_limit_with_timeline = max(
                1,
                max_results - len(state_link_candidates) - 1,
            )
            retained_primary = matches[:primary_limit_with_timeline]
            retained_primary_ids = [
                str(bucket.get("id") or "")
                for bucket in retained_primary
                if bucket.get("id")
            ]
            timeline_probe = _timeline_recall_neighbors(
                timeline_buckets,
                retained_primary_ids,
                query=recall_query,
                intent=intent_policy["intent"],
                world_filter=world_filter,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
                max_results=1,
                excluded_ids=(
                    set(retained_primary_ids)
                    | {
                        str(bucket.get("id") or "")
                        for bucket in state_link_candidates
                        if bucket.get("id")
                    }
                    | _session_seen_bucket_ids(timeline_buckets, session_id)
                    | _load_session_seen_ids(session_id)
                ),
            )
            if timeline_probe:
                timeline_slot_reserved = True
                timeline_fallback_matches = matches[primary_limit_with_timeline:]
                matches = retained_primary
        except Exception as exc:
            logger.warning(
                "Timeline preflight failed / 时间线预检失败: %s",
                type(exc).__name__,
            )
    with recall_stage("assembly"):
        set_recall_partial_result(_local_partial_recall_text(
            matches,
            max_results=max(0, max_results - len(state_link_candidates)),
            max_tokens=max_tokens,
            state_profile=state_profile,
        ))

    start_recall_stage("assembly")
    results = []
    token_used = 0
    result_buckets = []
    result_ids = []
    selected_content_fingerprints: set[str] = set()
    selected_e_evidence = []
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            # Only live chat beats (conversation/reflex) may trade one beat of
            # summary quality for latency — they retry next beat and sit under
            # the 11s injection deadline.  Explicit search (MCP breath, Y walk)
            # keeps the synchronous full summary and its read-only contracts.
            summary = await _dehydrate_for_recall(
                strip_wikilinks(bucket["content"]),
                clean_meta,
                bucket=bucket,
                allow_async_fallback=recall_policy in {"conversation", "reflex"},
            )
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            capture = _breath_candidate_capture.get()
            if isinstance(capture, list) and len(capture) < max_results:
                capture.append({"id": bucket["id"], "summary": summary})
            # Recall is read-only with respect to memory buckets.  Rendering a
            # search hit must not refresh last_active / activation_count or
            # trigger touch()'s bounded time ripple into neighboring buckets.
            if bucket.get("entity_match"):
                prefix = _recall_prefix(
                    bucket["id"],
                    "main",
                    "curated_rrf",
                    marker="[实体关联]",
                    bucket=bucket,
                    state_profile=state_profile,
                )
                summary = f"{prefix} {summary}"
            elif bucket.get("vector_match"):
                prefix = _recall_prefix(
                    bucket["id"],
                    "main",
                    "curated_rrf",
                    marker="[语义关联]",
                    bucket=bucket,
                    state_profile=state_profile,
                )
                summary = f"{prefix} {summary}"
            else:
                prefix = _recall_prefix(
                    bucket["id"],
                    "main",
                    "curated_rrf",
                    bucket=bucket,
                    state_profile=state_profile,
                )
                summary = f"{prefix} {summary}"
            results.append(summary)
            set_recall_partial_result("\n---\n".join(results))
            result_buckets.append(bucket)
            result_ids.append(bucket["id"])
            fingerprint = default_content_fingerprint(str(bucket.get("content") or ""))
            if fingerprint:
                selected_content_fingerprints.add(fingerprint)
            if bucket.get("_e_axis_annotation") is not None:
                selected_e_evidence.append((
                    bucket["_e_axis_annotation"],
                    float(bucket.get("_e_axis_resonance", 0.0) or 0.0),
                ))
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # X and Y may expand only the primary RRF results. State or side evidence
    # must not recursively become a timeline seed.
    main_result_ids = tuple(result_ids)

    # --- X-axis thread navigation: bounded previous/next supporting context ---
    # X consumes exactly the slot proven by preflight before Z/Y can claim it.
    # If rendering fails, primary fallback is restored before Z is assembled,
    # preserving the original primary-first selection and dedup semantics.
    remaining_timeline_slots = min(1, max(0, max_results - len(result_ids)))
    timeline_rendered = False
    if (
        timeline_slot_reserved
        and main_result_ids
        and remaining_timeline_slots
        and token_used < max_tokens
    ):
        timeline_msgs = []
        try:
            timeline_by_id = {
                str(bucket.get("id")): bucket
                for bucket in timeline_buckets
                if isinstance(bucket, dict) and bucket.get("id")
            }
            timeline_found = _timeline_recall_neighbors(
                timeline_buckets,
                main_result_ids,
                query=recall_query,
                intent=intent_policy["intent"],
                world_filter=world_filter,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
                max_results=remaining_timeline_slots,
                excluded_ids=(
                    set(result_ids)
                    | {
                        str(bucket.get("id") or "")
                        for bucket in state_link_candidates
                        if bucket.get("id")
                    }
                    | _session_seen_bucket_ids(timeline_buckets, session_id)
                    | _load_session_seen_ids(session_id)
                ),
            )
        except Exception as exc:
            logger.warning(
                "Timeline expansion failed / 时间线扩展失败: %s",
                type(exc).__name__,
            )
            timeline_found = []
            timeline_by_id = {}

        for timeline_neighbor in timeline_found:
            if token_used >= max_tokens or len(result_ids) >= max_results:
                break
            neighbor = timeline_by_id.get(timeline_neighbor.bucket_id)
            if not neighbor:
                continue
            try:
                fingerprint = default_content_fingerprint(
                    str(neighbor.get("content") or "")
                )
                if fingerprint and fingerprint in selected_content_fingerprints:
                    continue
                clean_meta = {
                    key: value
                    for key, value in neighbor["metadata"].items()
                    if key != "tags"
                }
                summary = await _dehydrate_for_recall(
                    strip_wikilinks(neighbor["content"]),
                    clean_meta,
                    bucket=neighbor,
                )
                summary_tokens = count_tokens_approx(summary)
                if token_used + summary_tokens > max_tokens:
                    break
                prefix = _recall_prefix(
                    timeline_neighbor.bucket_id,
                    "association",
                    "x_timeline",
                    relation=(
                        f"{timeline_neighbor.thread}:"
                        f"{timeline_neighbor.direction}:"
                        f"d{timeline_neighbor.distance}"
                        f"←{timeline_neighbor.via_id}"
                    ),
                    bucket=neighbor,
                    state_profile=state_profile,
                )
                timeline_msgs.append(f"{prefix} {summary}")
                token_used += summary_tokens
                result_buckets.append(neighbor)
                result_ids.append(timeline_neighbor.bucket_id)
                if fingerprint:
                    selected_content_fingerprints.add(fingerprint)
                capture = _breath_candidate_capture.get()
                if isinstance(capture, list) and len(capture) < max_results:
                    capture.append({
                        "id": timeline_neighbor.bucket_id,
                        "summary": summary,
                    })
            except Exception as exc:
                logger.warning(
                    "Failed to dehydrate timeline neighbor: %s",
                    type(exc).__name__,
                )
                continue
        if timeline_msgs:
            timeline_rendered = True
            results.append(
                "--- 时间线前后文（supporting only，不可替代主证据） ---\n"
                + "\n---\n".join(timeline_msgs)
            )
            set_recall_partial_result("\n---\n".join(results))

    # Preflight only proves structural adjacency.  If the neighbor cannot be
    # rendered (duplicate body, token budget, or dehydration error), restore
    # the primary candidates held back for the tentative X slot.  A failed X
    # expansion must be observationally equivalent to the pre-X path.
    if timeline_slot_reserved and not timeline_rendered:
        restored_main_messages = []
        restored_main_ids = []
        for bucket in timeline_fallback_matches:
            if token_used >= max_tokens or len(result_ids) >= max_results:
                break
            bucket_id = str(bucket.get("id") or "")
            if not bucket_id or bucket_id in result_ids:
                continue
            try:
                clean_meta = {
                    key: value
                    for key, value in bucket["metadata"].items()
                    if key != "tags"
                }
                if q_valence is not None and "valence" in clean_meta:
                    original_v = float(clean_meta.get("valence", 0.5))
                    shift = (q_valence - 0.5) * 0.2
                    clean_meta["valence"] = max(
                        0.0,
                        min(1.0, original_v + shift),
                    )
                summary = await _dehydrate_for_recall(
                    strip_wikilinks(bucket["content"]),
                    clean_meta,
                    bucket=bucket,
                )
                summary_tokens = count_tokens_approx(summary)
                if token_used + summary_tokens > max_tokens:
                    break
                marker = "[实体关联]" if bucket.get("entity_match") else (
                    "[语义关联]" if bucket.get("vector_match") else ""
                )
                prefix = _recall_prefix(
                    bucket_id,
                    "main",
                    "curated_rrf",
                    marker=marker,
                    bucket=bucket,
                    state_profile=state_profile,
                )
                restored_main_messages.append(f"{prefix} {summary}")
                restored_main_ids.append(bucket_id)
                token_used += summary_tokens
                result_buckets.append(bucket)
                result_ids.append(bucket_id)
                fingerprint = default_content_fingerprint(
                    str(bucket.get("content") or "")
                )
                if fingerprint:
                    selected_content_fingerprints.add(fingerprint)
                if bucket.get("_e_axis_annotation") is not None:
                    selected_e_evidence.append((
                        bucket["_e_axis_annotation"],
                        float(bucket.get("_e_axis_resonance", 0.0) or 0.0),
                    ))
                capture = _breath_candidate_capture.get()
                if isinstance(capture, list) and len(capture) < max_results:
                    capture.append({"id": bucket_id, "summary": summary})
            except Exception as exc:
                logger.warning(
                    "Failed to restore primary after timeline miss: %s",
                    type(exc).__name__,
                )
                continue
        if restored_main_messages:
            results.extend(restored_main_messages)
            main_result_ids = tuple((*main_result_ids, *restored_main_ids))
            set_recall_partial_result("\n---\n".join(results))

    # --- Z lifecycle state links: explicit current/history transition evidence ---
    # This is a bounded overlay over reviewed reciprocal links.  It does not
    # enter RRF, touch generic Y edges, or alter the underlying vector index.
    state_messages = []
    for state_bucket in state_link_candidates:
        if token_used >= max_tokens or len(result_ids) >= max_results:
            break
        state_bucket_id = str(state_bucket.get("id") or "")
        if not state_bucket_id or state_bucket_id in result_ids:
            continue
        try:
            fingerprint = default_content_fingerprint(
                str(state_bucket.get("content") or "")
            )
            if fingerprint and fingerprint in selected_content_fingerprints:
                continue
            clean_meta = {
                key: value
                for key, value in state_bucket["metadata"].items()
                if key != "tags"
            }
            summary = await _dehydrate_for_recall(
                strip_wikilinks(state_bucket["content"]),
                clean_meta,
                bucket=state_bucket,
            )
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            prefix = _recall_prefix(
                state_bucket_id,
                "state",
                "z_lifecycle",
                relation=str(state_bucket.get("_z_state_relation") or ""),
                bucket=state_bucket,
                state_profile=state_profile,
            )
            state_messages.append(f"{prefix} {summary}")
            token_used += summary_tokens
            result_buckets.append(state_bucket)
            result_ids.append(state_bucket_id)
            if fingerprint:
                selected_content_fingerprints.add(fingerprint)
            capture = _breath_candidate_capture.get()
            if isinstance(capture, list) and len(capture) < max_results:
                capture.append({"id": state_bucket_id, "summary": summary})
        except Exception as exc:
            logger.warning(
                "Failed to render Z state-link evidence: %s",
                type(exc).__name__,
            )
            continue
    if state_messages:
        results.append(
            "--- 状态链证据（按问题所问的现在/过去/变化使用） ---\n"
            + "\n---\n".join(state_messages)
        )

    # --- Typed relation expansion: bounded, bidirectional, at most 2 hops ---
    # --- Y 轴关系召回：只从实际展示的主结果出发，关联证据不进主排序 ---
    remaining_relation_slots = max(0, max_results - len(result_ids))
    relation_neighbor_cap = min(
        int(intent_policy.get("relation_neighbor_limit", 5)),
        remaining_relation_slots,
    )
    if (
        intent_policy["relation_depth"] >= 1
        and (main_result_ids if timeline_rendered else result_ids)
        and token_used < max_tokens
        and relation_neighbor_cap
    ):
        neighbor_msgs = []
        try:
            graph_buckets = await bucket_mgr.list_all(include_archive=False)
            graph_buckets = [
                bucket
                for bucket in graph_buckets
                if _is_main_recall_bucket(bucket)
            ]
            bucket_by_id = {
                str(bucket.get("id")): bucket
                for bucket in graph_buckets
                if isinstance(bucket, dict) and bucket.get("id")
            }
            relation_neighbors = _relation_recall_neighbors(
                graph_buckets,
                main_result_ids if timeline_rendered else result_ids,
                query=recall_query,
                intent=intent_policy["intent"],
                world_filter=world_filter,
                domain_filter=domain_filter,
                created_after=created_after,
                created_before=created_before,
                max_depth=intent_policy["relation_depth"],
                max_results=relation_neighbor_cap,
                excluded_ids=(
                    (
                        set(result_ids)
                        if timeline_rendered
                        else set()
                    )
                    | _session_seen_bucket_ids(graph_buckets, session_id)
                    | _load_session_seen_ids(session_id)
                ),
            )
        except Exception as exc:
            logger.warning("Relation graph expansion failed / 关系网扩展失败: %s", type(exc).__name__)
            relation_neighbors = []
            bucket_by_id = {}

        for relation_neighbor in relation_neighbors:
            if token_used >= max_tokens:
                break
            neighbor = bucket_by_id.get(relation_neighbor.bucket_id)
            if not neighbor:
                continue
            try:
                fingerprint = default_content_fingerprint(str(neighbor.get("content") or ""))
                if fingerprint and fingerprint in selected_content_fingerprints:
                    continue
                clean_meta = {
                    key: value
                    for key, value in neighbor["metadata"].items()
                    if key != "tags"
                }
                summary = await _dehydrate_for_recall(
                    strip_wikilinks(neighbor["content"]),
                    clean_meta,
                    bucket=neighbor,
                )
                summary_tokens = count_tokens_approx(summary)
                if token_used + summary_tokens > max_tokens:
                    break
                prefix = _recall_prefix(
                    relation_neighbor.bucket_id,
                    "association",
                    "y_relation",
                    relation=(
                        f"{relation_neighbor.relation_type}:"
                        f"{relation_neighbor.direction}:"
                        f"d{relation_neighbor.depth}"
                        f"←{relation_neighbor.via_id}"
                    ),
                    bucket=neighbor,
                    state_profile=state_profile,
                )
                neighbor_msgs.append(f"{prefix} {summary}")
                token_used += summary_tokens
                result_buckets.append(neighbor)
                result_ids.append(relation_neighbor.bucket_id)
                if fingerprint:
                    selected_content_fingerprints.add(fingerprint)
            except Exception as exc:
                logger.warning(
                    "Failed to dehydrate relation neighbor / 关系邻居脱水失败: %s",
                    type(exc).__name__,
                )
                continue
        if neighbor_msgs:
            results.append(
                "--- 关系网关联旁证（supporting only，不可替代主证据） ---\n"
                + "\n---\n".join(neighbor_msgs)
            )

    # --- E-axis independent resonance channel (bounded supporting evidence) ---
    # Only explicit emotional wording/coordinates unlock E-only memories.  A
    # neutral query can still use E to break close topical ties above, but it
    # cannot pull an unrelated emotional memory into the prompt.
    if (
        e_recall_cfg is not None
        and e_query_emotion is not None
        and e_query_emotion.explicit
        and e_recall_cfg.side_channel_limit > 0
        and token_used < max_tokens
    ):
        e_side_messages = []
        excluded_e_ids = set(result_ids) | _load_session_seen_ids(session_id)
        prelim_ids = rank_annotation_bucket_ids(
            e_rows_by_bucket,
            e_query_emotion,
            limit=e_recall_cfg.side_channel_scan_limit,
        )
        for e_bucket_id in prelim_ids:
            if timeline_rendered and len(result_ids) >= max_results:
                break
            if len(e_side_messages) >= e_recall_cfg.side_channel_limit:
                break
            if e_bucket_id in excluded_e_ids:
                continue
            try:
                e_bucket = await bucket_mgr.get(e_bucket_id)
                if not e_bucket or not _passes_nonkeyword_recall_filters(
                    e_bucket,
                    world_filter_set=wf_set,
                    domain_filter=domain_filter,
                    created_after=created_after,
                    created_before=created_before,
                ):
                    continue
                if not _filter_session_seen([e_bucket], session_id):
                    continue
                fingerprint = default_content_fingerprint(str(e_bucket.get("content") or ""))
                if fingerprint and fingerprint in selected_content_fingerprints:
                    continue
                if not _filter_z_fact_candidates(
                    [e_bucket],
                    query=recall_query,
                    intent=intent_policy["intent"],
                ):
                    continue
                e_annotation = select_current_annotation(
                    e_rows_by_bucket.get(e_bucket_id, ()),
                    e_bucket,
                    e_recall_cfg,
                )
                if e_annotation is None:
                    continue
                e_resonance = e_axis_resonance_score(
                    e_query_emotion,
                    e_annotation,
                )
                if e_resonance < e_recall_cfg.side_channel_min_resonance:
                    continue
                clean_meta = {
                    key: value
                    for key, value in e_bucket["metadata"].items()
                    if key != "tags"
                }
                summary = await _dehydrate_for_recall(
                    strip_wikilinks(e_bucket["content"]),
                    clean_meta,
                    bucket=e_bucket,
                )
                summary_tokens = count_tokens_approx(summary)
                if token_used + summary_tokens > max_tokens:
                    break
                prefix = _recall_prefix(
                    e_bucket_id,
                    "side_channel",
                    "e_emotion",
                    marker="[情绪共鸣]",
                    bucket=e_bucket,
                    state_profile=state_profile,
                )
                e_side_messages.append(
                    f"{prefix} [resonance:{e_resonance:.3f}] {summary}"
                )
                token_used += summary_tokens
                result_buckets.append(e_bucket)
                result_ids.append(e_bucket_id)
                if fingerprint:
                    selected_content_fingerprints.add(fingerprint)
                excluded_e_ids.add(e_bucket_id)
                selected_e_evidence.append((e_annotation, e_resonance))
            except Exception as exc:
                logger.warning(
                    "E resonance candidate skipped after validation: %s",
                    type(exc).__name__,
                )
                continue
        if e_side_messages:
            results.append(
                "--- E轴情绪共鸣旁证（supporting experience only，"
                "不可替代事实） ---\n"
                + "\n---\n".join(e_side_messages)
            )

    # --- Random surfacing is opt-in after PR-1 noise reduction.
    # --- 减噪后随机漂浮改为显式配置，默认关闭，避免检索不足时硬塞旧噪音。
    random_cfg = config.get("random_surfacing", {}) or {}
    try:
        random_chance = float(random_cfg.get("search_underflow_chance", 0.0) or 0.0)
    except (TypeError, ValueError):
        random_chance = 0.0
    if (
        recall_policy == "search"
        and len(matches) < 3
        and random_chance > 0
        and random.random() < random_chance
    ):
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            all_buckets = [
                bucket
                for bucket in all_buckets
                if _is_main_recall_bucket(bucket)
            ]
            matched_ids = set(result_ids)
            seen_ids = (
                _session_seen_bucket_ids(all_buckets, session_id)
                | _load_session_seen_ids(session_id)
            )
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and b["id"] not in seen_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
                and (wf_set is None or world_matches(b["metadata"].get("world", ""), wf_set))
            ]
            low_weight = _filter_z_fact_candidates(
                low_weight,
                query=recall_query,
                intent=intent_policy["intent"],
            )
            if low_weight:
                remaining_slots = max(0, max_results - len(result_ids))
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight), remaining_slots))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await _dehydrate_for_recall(strip_wikilinks(b["content"]), clean_meta, bucket=b)
                    prefix = _recall_prefix(
                        b["id"],
                        "side_channel",
                        "random_surface",
                        bucket=b,
                        state_profile=state_profile,
                    )
                    drift_results.append(f"[surface_type: random] {prefix}\n{summary}")
                    result_buckets.append(b)
                    result_ids.append(b["id"])
                if drift_results:
                    results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        finish_recall_stage("assembly")
        return _append_body_state_block(
            "未找到相关记忆。",
            [],
            session_id,
            include_body_state,
            reset_body_state,
        )

    text = "\n---\n".join(results)
    if e_recall_cfg is not None and selected_e_evidence:
        e_posture = derive_response_posture(selected_e_evidence)
        if e_posture is not None:
            text = (
                text
                + "\n---\n"
                + format_response_posture(
                    e_posture,
                    activation_id=e_recall_cfg.activation_id,
                )
            )
    text = _append_body_state_block(
        text,
        result_buckets,
        session_id,
        include_body_state,
        reset_body_state,
    )
    _remember_session_seen_ids(session_id, result_ids)
    finish_recall_stage("assembly")
    return await _tool_result_with_optional_images(text, result_buckets, include_images)


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    source_bucket: str = "",
    valence: float = -1,
    arousal: float = -1,
    image_base64: str = "",
    image_filename: str = "image",
    world: str = "",
    chord_tag: str = "",
    domain: str = "",
) -> str:
    """存储单条记忆,自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现)。source_bucket=被消化的记忆桶ID(feel模式下,标记源记忆为已消化)。image_base64=可选,base64编码的图片数据,会上传到R2并把URL插入正文(允许此条记忆带图)。image_filename=图片名称提示(默认image)。world=显式指定世界归属,留空时走全局current_world(日常聊天=空,角色扮演=具体世界名),"通用"表示跨世界设定。feel桶不归属世界。chord_tag=可选和弦记号串(如"Em(maj7) → A13#11 · 92bpm · f"),作为情绪色调索引,只用于跨窗口标记,不参与表达。紧张系和弦(m(maj7)/♭9/dim等)加动作词disambiguator(盯/压/憋/狂),一行最多4个和弦,段落切换用"; "分隔,详见 INTERNALS.md 5.12。feel桶不打chord_tag。merge时若新带chord_tag会覆盖旧桶。domain=显式指定主题域(csv),非空时override dehydrator 自动推断,用于跨 Agent 工程日志隔离(如 hajimi-工程)。feel/pinned 路径同样适用。"""
    await _ensure_decay_background()
    _maybe_start_backfill()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Resolve effective world / 解析当前桶的 world 归属 ---
    # 显式传 world > 全局 current_world。feel 桶在下面单独处理（feel 跨世界）。
    effective_world = (world or "").strip() or (config.get("current_world", "") or "").strip()

    # --- Optional image upload to R2 / 可选：上传图片到 R2 ---
    # If image_base64 provided and R2 configured, upload and prepend URL
    # markdown to content so the image is rendered in Obsidian and
    # surfaced when this bucket is read later.
    # 若提供了 image_base64 且 R2 已配置，上传并在正文前部插入图片 URL，
    # 这样 Obsidian 能直接渲染，桶被读取时图片 URL 也会跟着 content 出来。
    image_url: str = ""
    if image_base64 and image_base64.strip():
        try:
            image_url = r2_storage.upload_base64(image_base64, image_filename) or ""
        except Exception as e:
            logger.warning(f"R2 image upload raised / R2 上传抛错: {e}")
            image_url = ""
        if image_url:
            # Prepend image markdown so dehydrator/Obsidian both see it
            # 在正文前插入图片 markdown，dehydrator 和 Obsidian 都能识别
            content = f"![{image_filename}]({image_url})\n\n{content}"
            logger.info(f"Hold attached image / 附加图片: {image_url}")
        else:
            logger.warning(
                "Image was provided but R2 upload returned no URL "
                "(R2 disabled or upload failed) / "
                "提供了图片但 R2 上传未返回 URL（R2 未启用或上传失败）"
            )

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        # 2026-07-12 修：原先硬编码空 tags/domain，调用方传的字段全丢，
        # imprint 写进来的记忆全变'未分类'。改用调用方实际传入的值。
        feel_domain = [d.strip() for d in (domain or "").split(",") if d.strip()]
        async with bucket_mgr._maintenance_barrier.shared_async():
            bucket_id = await bucket_mgr.create(
                content=content,
                tags=extra_tags,
                importance=5,
                domain=feel_domain,
                valence=feel_valence,
                arousal=feel_arousal,
                name=None,
                bucket_type="feel",
            )
            try:
                await embedding_engine.generate_and_store(bucket_id, content)
            except Exception:
                pass
            # --- Mark source memory as digested + store model's valence perspective ---
            # --- 标记源记忆为已消化 + 存储模型视角的 valence ---
            if source_bucket and source_bucket.strip():
                try:
                    update_kwargs = {"digested": True}
                    if 0 <= valence <= 1:
                        update_kwargs["model_valence"] = feel_valence
                    await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
                except Exception as e:
                    logger.warning(f"Failed to mark source as digested / 标记已消化失败: {e}")
        await _synchronize_bucket_entities(bucket_id, content)
        _mark_briefing_cache_dirty("hold_feel")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "", "entities": [],
        }

    # 显式 domain override（用于跨 Agent 工程日志隔离，如 hajimi-工程）
    # 留空走 dehydrator 自动推断
    explicit_domain = [d.strip() for d in (domain or "").split(",") if d.strip()]
    domain = explicit_domain or analysis["domain"]
    valence = analysis["valence"]
    arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        async with bucket_mgr._maintenance_barrier.shared_async():
            bucket_id = await bucket_mgr.create(
                content=content,
                tags=all_tags,
                importance=10,
                domain=domain,
                valence=valence,
                arousal=arousal,
                name=suggested_name or None,
                retrieval_keys=_entity_retrieval_keys(
                    analysis.get("entities", [])
                ),
                bucket_type="permanent",
                pinned=True,
                world=effective_world,
                chord_tag=chord_tag,
            )
            try:
                await embedding_engine.generate_and_store(bucket_id, content)
            except Exception:
                pass
        await _synchronize_bucket_entities(
            bucket_id, content, analysis.get("entities", [])
        )
        _mark_briefing_cache_dirty("hold_pinned")
        return f"📌钉选→{bucket_id} {','.join(domain)}"

    # --- Step 2: merge or create / 合并或新建 ---
    bucket_id, result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        world=effective_world,
        chord_tag=chord_tag,
        entities=analysis.get("entities", []),
    )
    _mark_briefing_cache_dirty("hold_merge" if is_merged else "hold_create")

    # --- Step 3: safe relations write directly; dangerous ones stay pending ---
    relation_proposals: list[dict] = []
    if not is_merged:
        try:
            relation_proposals = await _auto_infer_edges(
                source_id=bucket_id, content=content, world=effective_world
            )
        except Exception as e:
            logger.warning(f"Auto-edge inference failed / 自动建边失败: {e}")

    action = "合并→" if is_merged else "新建→"
    base = f"{action}{result_name} {','.join(domain)}"
    if not relation_proposals:
        return base
    applied = sum(e.get("status") == "applied" for e in relation_proposals)
    pending = sum(e.get("status") == "pending_review" for e in relation_proposals)
    related_lines = [
        f"  • [{'已落图' if e.get('status') == 'applied' else '待审'}] "
        f"[{e['type']}] {e['target_name']} ({e['target']})"
        + (f" — {e['note']}" if e.get("note") else "")
        for e in relation_proposals
    ]
    return (
        f"{base} +{applied}条安全关系已落图，+{pending}条危险关系待审\n"
        "候选关联：\n" + "\n".join(related_lines)
    )


# =============================================================
# Tool: experience — primary-agent-authored E record
# =============================================================
@mcp.tool()
async def experience(
    content: str,
    e_authored_by: str,
    e_initial_priority: int,
    e_valence: float,
    e_arousal: float,
    e_tension: float,
    e_response_tendency: str,
    e_growth_delta: str,
    e_confidence: float = 1.0,
    source_bucket_id: str = "",
    proposal_key: str = "",
    name: str = "",
    domain: str = "关系",
    world: str = "",
) -> str:
    """主对话 AI 亲自写 E 轴体验并选择 1..100 初始优先级。

    内容原样落桶，不经过 scorer/digest 改写。模型夜跑只能产生 e_proposal；
    传 proposal_key 可在写成权威 E 后把对应提案标为 reviewed。
    """
    if not content or not content.strip():
        return "E 内容为空，未写入。"
    author = str(e_authored_by or "").strip()
    source_id = str(source_bucket_id or "").strip()
    proposal_id = str(proposal_key or "").strip()
    if not author:
        return "e_authored_by 必填；E 必须由主对话 AI 具名书写。"
    if proposal_id:
        proposal = _get_review_queue().get(proposal_id)
        if proposal is None or proposal.get("kind") != KIND_E_PROPOSAL:
            return "未找到对应的 E proposal。"
        for bucket in await bucket_mgr.list_all(include_archive=True):
            metadata = bucket.get("metadata") or {}
            if metadata.get("e_proposal_key") == proposal_id:
                return (
                    f"E→{bucket['id']} [author:{metadata.get('e_authored_by')}] "
                    f"[initial_priority:{metadata.get('e_initial_priority')}] "
                    "[idempotent:existing]"
                )
        if proposal.get("status") != "pending":
            return "对应的 E proposal 已裁决，未重复写入。"
        proposed_source = str(proposal.get("source_bucket_id") or "")
        if source_id and source_id != proposed_source:
            return "source_bucket_id 与 E proposal 不一致。"
        source_id = proposed_source
    if source_id and await bucket_mgr.get(source_id) is None:
        return f"E 来源桶不存在: {source_id}"
    effective_world = (world or "").strip() or (
        config.get("current_world", "") or ""
    ).strip()
    domains = [
        part.strip()
        for part in str(domain or "").split(",")
        if part.strip()
    ]
    try:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=["lmc5", "experience", "relationship_moment"],
            importance=max(
                1,
                min(10, (int(e_initial_priority) + 9) // 10),
            ),
            domain=domains or ["关系"],
            valence=max(
                0.0,
                min(1.0, (float(e_valence) + 1.0) / 2.0),
            ),
            arousal=float(e_arousal),
            name=name or "主AI体验",
            bucket_type="dynamic",
            world=effective_world,
            actor=f"e-axis:{author}",
            e_authored_by=author,
            e_initial_priority=e_initial_priority,
            e_valence=e_valence,
            e_arousal=e_arousal,
            e_tension=e_tension,
            e_confidence=e_confidence,
            e_response_tendency=e_response_tendency,
            e_growth_delta=e_growth_delta,
            e_source_bucket_id=source_id,
            e_proposal_key=proposal_id,
        )
    except Exception as exc:
        logger.warning(
            "Primary-authored E write rejected: %s",
            type(exc).__name__,
        )
        return f"E 写入失败: {exc}"
    try:
        vector_ready = await embedding_engine.generate_and_store(
            bucket_id,
            content,
        )
    except Exception as exc:
        vector_ready = False
        logger.warning(
            "Primary-authored E vector unavailable for %s: %s",
            bucket_id,
            type(exc).__name__,
        )
    if proposal_id:
        try:
            _get_review_queue().resolve(
                proposal_id,
                "reviewed",
                reviewer=author,
                verdict_note=f"primary-authored E bucket {bucket_id}",
            )
        except Exception as exc:
            logger.warning(
                "E proposal remained pending after authored write %s: %s",
                bucket_id,
                type(exc).__name__,
            )
    _mark_briefing_cache_dirty("experience_create")
    return (
        f"E→{bucket_id} [author:{author}] "
        f"[initial_priority:{e_initial_priority}] "
        f"[vector:{'ready' if vector_ready else 'missing'}]"
    )


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str, world: str = "", chord_tag: str = "") -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。world留空走全局current_world。chord_tag=可选和弦记号串作为整段日记的色调,会打到所有子桶上(子桶共用同一色调)。"""
    _grow_t0 = time.perf_counter()
    _content_len = len(content.strip()) if content else 0
    await _ensure_decay_background()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Resolve effective world / 解析当前批次的 world 归属 ---
    effective_world = (world or "").strip() or (config.get("current_world", "") or "").strip()

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            prepared_content = await dehydrator.ensure_self_contained(
                content.strip(),
                source_context=content.strip(),
            )
        except SelfContainmentError as exc:
            logger.warning("grow short-content rejected: %s", exc)
            return "内容含无法唯一确认的指代，未写入记忆。请补充具体人名、地点或项目后重试。"
        try:
            _t_a = time.perf_counter()
            analysis = await dehydrator.analyze(prepared_content)
            _ela_a = time.perf_counter() - _t_a
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "", "entities": [],
            }
            _ela_a = time.perf_counter() - _t_a
        _t_m = time.perf_counter()
        _bid, result_name, is_merged = await _merge_or_create(
            content=prepared_content,
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
            world=effective_world,
            chord_tag=chord_tag,
            require_self_contained=True,
            recall_before_write=True,
            entities=analysis.get("entities", []),
        )
        _ela_m = time.perf_counter() - _t_m
        _mark_briefing_cache_dirty(
            "grow_fast_merge" if is_merged else "grow_fast_create"
        )
        _ela_total = time.perf_counter() - _grow_t0
        logger.info(
            f"grow.timing path=fast chars={_content_len} "
            f"analyze={_ela_a:.2f}s merge={_ela_m:.2f}s total={_ela_total:.2f}s "
            f"action={'merged' if is_merged else 'created'}"
        )
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        _t_d = time.perf_counter()
        # Long-form grow must not discard the incoming memory merely because
        # self-containment arbitration is unavailable or inconclusive.  The
        # dehydrator keeps empty content and credentials fail-closed.
        _unresolved_sink: list = []
        items = await dehydrator.digest(
            content,
            fail_open=True,
            unresolved_sink=_unresolved_sink,
        )
        if _unresolved_sink:
            logger.info("grow digest fail-open: %s", _unresolved_sink[:5])
        _ela_d = time.perf_counter() - _t_d
    except SelfContainmentError as e:
        logger.warning(f"Diary digest self-containment failed / 指代消解失败: {e}")
        logger.info(
            f"grow.timing path=long chars={_content_len} self_containment_failed=1 "
            f"total={time.perf_counter() - _grow_t0:.2f}s"
        )
        return "内容含无法唯一确认的指代，整批未写入。请补充具体人名、地点或项目后重试。"
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败: {e}")
        logger.info(
            f"grow.timing path=long chars={_content_len} digest_failed=1 "
            f"total={time.perf_counter() - _grow_t0:.2f}s"
        )
        return f"日记整理失败: {e}"

    if not items:
        logger.info(
            f"grow.timing path=long chars={_content_len} digest={_ela_d:.2f}s items=0 "
            f"total={time.perf_counter() - _grow_t0:.2f}s"
        )
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0
    _item_elapsed = []  # per-item merge_or_create elapsed seconds

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for _i, item in enumerate(items):
        _t_i = time.perf_counter()
        try:
            _bid, result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
                world=effective_world,
                chord_tag=chord_tag,
                require_self_contained=True,
                recall_before_write=True,
                entities=item.get("entities", []),
            )

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name') or result_name}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")
        finally:
            _item_elapsed.append(time.perf_counter() - _t_i)

    _ela_total = time.perf_counter() - _grow_t0
    _items_sum = sum(_item_elapsed)
    _items_max = max(_item_elapsed) if _item_elapsed else 0.0
    logger.info(
        f"grow.timing path=long chars={_content_len} digest={_ela_d:.2f}s "
        f"items={len(items)} items_sum={_items_sum:.2f}s items_max={_items_max:.2f}s "
        f"created={created} merged={merged} total={_ela_total:.2f}s"
    )

    if created or merged:
        _mark_briefing_cache_dirty("grow_batch")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    pinned: int = -1,
    protected: int = -1,
    digested: int = -1,
    content: str = "",
    world: str = "",
    chord_tag: str = "",
    delete: bool = False,
    add_relation: str = "",
    remove_relation: str = "",
) -> str:
    """修改记忆元数据或内容。resolved=1沉底/0激活,pinned=1钉选/0取消,protected=1永不遗忘(无条件入简报)/0摘保护回归衰减池(仅清过时工程桶,感情桶绝不摘),digested=1隐藏(保留但不浮现)/0取消隐藏,content=替换桶正文,delete=True删除。world=改世界归属(传"(none)"清空回日常),只传需改的,-1或空=不改。chord_tag=改情绪色调和弦串(传"(none)"清空),空=不改。add_relation格式"type:target_id"或"type:target_id:note",6类:causes/contributes/improves/explains/updates/kin。remove_relation格式"target_id"或"type:target_id"。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        async with bucket_mgr._maintenance_barrier.shared_async():
            success = await bucket_mgr.delete(bucket_id)
            if success:
                embedding_engine.delete_embedding(bucket_id)
        if success:
            _unlink_bucket_entities(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    if pinned in (0, 1):
        updates["pinned"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10  # pinned → lock importance
    if protected in (0, 1):
        # protected=1 永不自动遗忘（无条件入简报核心区）；=0 摘掉保护、回归正常衰减池。
        # 仅用于清理过时工程态桶——感情/约定/纪念日域桶绝不摘（摘了违 5.14 不回避感情约定）。
        updates["protected"] = bool(protected)
    if digested in (0, 1):
        updates["digested"] = bool(digested)
    if world:
        w = world.strip()
        # sentinel "(none)" → 清空 world 字段（挪回日常）
        updates["world"] = "" if w == "(none)" else w
    if chord_tag:
        ct = chord_tag.strip()
        # sentinel "(none)" → 清空 chord_tag 字段
        updates["chord_tag"] = "" if ct == "(none)" else ct
    if content:
        updates["content"] = content

    # --- Relation edits / 关系边操作（独立于 metadata update，避免和 update 字段竞争）---
    relation_msgs = []
    if add_relation and add_relation.strip():
        parts = [p.strip() for p in add_relation.split(":", 2)]
        if len(parts) < 2:
            return "add_relation 格式错误，需 'type:target_id' 或 'type:target_id:note'。"
        rel_type, target_id = parts[0], parts[1]
        note = parts[2] if len(parts) >= 3 else ""
        ok = await bucket_mgr.add_relation(bucket_id, target_id, rel_type, note)
        relation_msgs.append(f"+边 [{rel_type}→{target_id}]" if ok else f"加边失败 [{rel_type}→{target_id}]")
    if remove_relation and remove_relation.strip():
        parts = [p.strip() for p in remove_relation.split(":", 1)]
        if len(parts) == 1:
            rel_type, target_id = "", parts[0]
        else:
            rel_type, target_id = parts[0], parts[1]
        n = await bucket_mgr.remove_relation(bucket_id, target_id, rel_type)
        relation_msgs.append(f"-边 ×{n} [{target_id}]" if n else f"删边未命中 [{target_id}]")

    if not updates and not relation_msgs:
        return "没有任何字段需要修改。"

    if updates:
        try:
            async with bucket_mgr._maintenance_barrier.shared_async():
                success = await bucket_mgr.update(bucket_id, **updates)
                if success and "content" in updates:
                    try:
                        await embedding_engine.generate_and_store(
                            bucket_id,
                            updates["content"],
                        )
                    except Exception as e:
                        logger.warning(f"Embedding refresh after update failed / 改内容后向量刷新失败: {bucket_id}: {e}")
        except ResolvedGuardError as e:
            return f"❌ 守卫拦截: {e}。这条铁律是 5.10 黑洞修复后落代码的兜底——保护域桶=持续状态，不该有'完结'。"
        if success and "content" in updates:
            await _synchronize_bucket_entities(bucket_id, updates["content"])
        if not success:
            return f"修改失败: {bucket_id}"
    else:
        success = True

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    if "digested" in updates:
        if updates["digested"]:
            changed += " → 已隐藏，保留但不再浮现"
        else:
            changed += " → 已取消隐藏，重新参与浮现"
    if relation_msgs:
        changed = (changed + "; " if changed else "") + "; ".join(relation_msgs)
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool: inspect — view full bucket content by ID
# 工具：inspect — 按 ID 查看记忆桶完整内容（不脱水）
# Bypasses surfacing/search; for engineering ops (merge, edit, audit)
# where the caller already knows the ID and needs to see the raw content.
# 绕过浮现/检索；用于已知 ID、需要看原文的工程操作（整合、编辑、审查）。
# =============================================================
@mcp.tool()
async def inspect(bucket_id: str) -> str:
    """按 ID 查看记忆桶完整内容（不脱水）。用于整合/编辑/审查时需看原文的工程操作。"""
    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    bucket = await bucket_mgr.get(bucket_id.strip())
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    meta = bucket.get("metadata", {})
    content = strip_wikilinks(bucket.get("content", ""))

    try:
        score = decay_engine.calculate_score(meta)
    except Exception:
        score = 0.0

    name = meta.get("name") or "(未命名)"
    domains = ",".join(meta.get("domain", []) if isinstance(meta.get("domain"), list) else [str(meta.get("domain", ""))])
    tags = ",".join(meta.get("tags", []) if isinstance(meta.get("tags"), list) else [str(meta.get("tags", ""))])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.5)
    imp = meta.get("importance", "?")
    world = meta.get("world", "") or "(日常)"
    chord = meta.get("chord_tag", "") or ""
    flags = []
    if meta.get("pinned"): flags.append("pinned")
    if meta.get("protected"): flags.append("protected")
    if meta.get("resolved"): flags.append("resolved")
    if meta.get("digested"): flags.append("digested")
    if meta.get("type"): flags.append(f"type={meta['type']}")
    flag_str = ", ".join(flags) if flags else "无"

    header = (
        f"[bucket_id:{bucket['id']}] {name}\n"
        f"主题: {domains}  标签: {tags}\n"
        f"情感: V{val:.1f}/A{aro:.1f}  重要性: {imp}  当前分: {score:.2f}\n"
        f"world: {world}  标志: {flag_str}\n"
        + (f"chord_tag: {chord}\n" if chord else "")
        + f"创建: {meta.get('created_at', '?')}  更新: {meta.get('updated_at', '?')}"
    )

    relations = meta.get("relations") or []
    rel_lines = []
    if isinstance(relations, list):
        for r in relations:
            if isinstance(r, dict):
                note = r.get("note", "")
                note_str = f" ({note})" if note else ""
                rel_lines.append(f"  - {r.get('type', '?')} → {r.get('target', '?')}{note_str}")
    rel_block = ("\n\n关系边:\n" + "\n".join(rel_lines)) if rel_lines else ""

    return f"{header}\n\n--- 正文 ---\n{content}{rel_block}"


# =============================================================
# Tool: update_bucket — 改桶正文/元数据,用于事实订正/合并/清整
# 受 ResolvedGuard 限制(feel-type / 保护域桶禁止 resolved=1)。
# 改 pinned/type/domain 会触发桶文件移动到对应目录。
# =============================================================
@mcp.tool()
async def update_bucket(
    bucket_id: str,
    content: str = "",
    chord_tag: str = "",
    name: str = "",
) -> str:
    """按 ID 改桶正文/元数据。content=新正文(空字符串=不改);chord_tag=色调记号串(空=不改);name=桶名(空=不改)。返回处理结果。"""
    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    kwargs: dict = {}
    if content:
        kwargs["content"] = content
    if chord_tag:
        kwargs["chord_tag"] = chord_tag
    if name:
        kwargs["name"] = name

    if not kwargs:
        return "至少要传一个改动 (content / chord_tag / name)。"

    ok = await _apply_bucket_update(
        bucket_id.strip(),
        kwargs,
        actor="mcp:update_bucket",
    )

    if not ok:
        return f"桶不存在或改写失败: {bucket_id}"
    return f"已更新桶 {bucket_id}: {', '.join(kwargs.keys())}"


# =============================================================
# Tool: delete_bucket — 删桶(不可恢复)
# 受保护桶(protected=True)拒删。需 confirm=True 防误删。
# =============================================================
@mcp.tool()
async def delete_bucket(bucket_id: str, confirm: bool = False) -> str:
    """按 ID 删桶(不可恢复)。需传 confirm=True 才执行(防误删)。受保护桶拒删。"""
    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"
    if not confirm:
        return "需 confirm=True 才执行删除(防误删)。"

    try:
        ok = await bucket_mgr.delete(bucket_id.strip())
    except Exception as e:
        return f"删桶失败: {e}"

    if not ok:
        return f"桶不存在 / 受保护 / 文件删除失败: {bucket_id}"
    _unlink_bucket_entities(bucket_id.strip())
    return f"已删除桶 {bucket_id}"


# =============================================================
# Tool: backfill_relations — propose relations for existing buckets
# 工具：backfill_relations — 给老桶批量生成待审关系提议
# Hold-time auto-edge only fires on new buckets; this tool fills in the
# graph for memories that existed before the feature shipped. Batched to
# avoid MCP timeout and to let the caller resume between calls.
# =============================================================
@mcp.tool()
async def backfill_relations(
    bucket_id: str = "",
    limit: int = 5,
    offset: int = 0,
) -> str:
    """对已有桶推断关系：安全边审计落图，危险边具名待审。
    bucket_id=指定单桶处理（最快验证用）。
    bucket_id 为空时按 limit/offset 批量遍历 dynamic 桶（跳过 pinned/permanent/feel/resolved），每次最多 10 个，多次调用滚动跑完。
    返回每桶生成了几条提议和下一批 offset。"""
    if bucket_id and bucket_id.strip():
        bucket = await bucket_mgr.get(bucket_id.strip())
        if not bucket:
            return f"未找到桶: {bucket_id}"
        try:
            edges = await _auto_infer_edges(
                source_id=bucket["id"],
                content=bucket["content"],
                world=bucket["metadata"].get("world", ""),
            )
            applied = sum(e.get("status") == "applied" for e in edges)
            pending = sum(e.get("status") == "pending_review" for e in edges)
            return f"{bucket['id']}: +{applied}安全边已落图，+{pending}危险边待审"
        except Exception as e:
            logger.warning(f"backfill single failed {bucket_id}: {e}")
            return f"{bucket_id}: 失败 {e}"

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        return f"列桶失败: {e}"

    # 跳过钉选/保护/permanent/feel/resolved——这些桶要么不需要长边、要么是潜意识素材库
    eligible = [
        b for b in all_buckets
        if not b["metadata"].get("pinned")
        and not b["metadata"].get("protected")
        and b["metadata"].get("type") not in ("feel", "permanent")
        and not b["metadata"].get("resolved", False)
    ]
    eligible.sort(key=lambda b: b["id"])

    limit = max(1, min(int(limit), 10))
    offset = max(0, int(offset))
    batch = eligible[offset:offset + limit]

    if not batch:
        return f"无桶可处理 (eligible={len(eligible)}, offset={offset})"

    results = []
    total_applied = 0
    total_pending = 0
    for b in batch:
        try:
            edges = await _auto_infer_edges(
                source_id=b["id"],
                content=b["content"],
                world=b["metadata"].get("world", ""),
            )
            applied = sum(e.get("status") == "applied" for e in edges)
            pending = sum(e.get("status") == "pending_review" for e in edges)
            results.append(f"{b['id'][:6]}+{applied}a/{pending}r")
            total_applied += applied
            total_pending += pending
        except Exception as e:
            logger.warning(f"backfill bucket {b['id']} failed: {e}")
            results.append(f"{b['id'][:6]}!err")

    next_offset = offset + len(batch)
    remaining = len(eligible) - next_offset
    return (
        f"批 {offset}-{next_offset - 1}/{len(eligible)} | "
        f"+{total_applied}安全边已落图/+{total_pending}危险边待审 | {' '.join(results)} | "
        f"剩 {remaining}, next offset={next_offset}"
    )


# =============================================================
# Tool: switch_world — change global current_world pointer at runtime
# 工具：switch_world — 切换全局当前世界指针
# =============================================================
@mcp.tool()
async def switch_world(world: str = "") -> str:
    """切换全局当前世界。空字符串=日常模式(只浮现日常+通用桶),具体世界名=角色扮演模式。
    生效范围：之后所有 hold(不传 world 时)写到该世界,breath(不传 world 时)只浮该世界+通用。
    持久化到 {buckets_dir}/.ombre_runtime.yaml,重启不丢。pulse 可看当前指针。"""
    target = (world or "").strip()
    valid_worlds = config.get("worlds", []) or []
    if target and target not in valid_worlds:
        return (
            f"未知世界: {target!r}。已知 worlds: {valid_worlds}\n"
            f"如要新增世界,先在 config.yaml 的 worlds: 列表里加上,再切换。"
        )
    try:
        # 扫盘 #12：同步文件 I/O 挪 to_thread，NAS 磁盘慢时不卡 event loop
        await asyncio.to_thread(save_current_world, config["buckets_dir"], target)
    except OSError as e:
        return f"持久化失败: {e}"
    config["current_world"] = target
    # 同步涩涩目录加载开关：切进"涩涩"才扫那个文件夹，切出即物理隔离
    try:
        bucket_mgr.nsfw_active = (target == "涩涩")
        bucket_mgr.invalidate_list_all_cache()
    except Exception:
        pass
    _register_briefing_profile((1000, "", False, "", "text", target))
    _register_briefing_profile((1500, "", False, "", "json", target))
    _mark_briefing_cache_dirty("switch_world")
    label = target if target else "日常模式 (空)"
    logger.info(f"current_world switched → {label}")
    return f"已切换到 → {label}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def review_pending(kind: str = "") -> str:
    """只读列出「待审队列」里的 pending 候选——M 轴巡检建议、机器自动推断的危险
    关系边（#3 关系闸）和事实演化冲突（#2 Z轴），都先挂这等人显式裁决。

    永不改库：本工具只把清单念出来，建边/supersede/resolve 都需人另外显式操作。
    kind 可选过滤：'metabolism' / 'relation' / 'z_conflict' / 'e_proposal' / 留空看全部。
    """
    k = (kind or "").strip().lower()
    allowed_kinds = (
        KIND_METABOLISM,
        KIND_RELATION,
        KIND_Z_CONFLICT,
        KIND_E_PROPOSAL,
    )
    if k and k not in allowed_kinds:
        return (
            "kind 只能是 "
            f"'{KIND_METABOLISM}' / '{KIND_RELATION}' / "
            f"'{KIND_Z_CONFLICT}' / '{KIND_E_PROPOSAL}' 或留空。"
        )
    try:
        # The queue is a small local JSONL ledger.  Reading it inline avoids
        # depending on the process-wide default executor, which can be
        # unavailable during MCP shutdown and made this read-only tool hang.
        items = _get_review_queue().list_pending(k or None)
    except Exception as e:
        return f"读取待审队列失败: {e}"
    return _render_review_md(items)


@mcp.tool()
async def pulse(include_archive: bool = False, full: bool = False, limit: int = 40) -> str:
    """系统状态+记忆桶导航。默认按权重只显示 Top-`limit` 个桶的目录化摘要(防止记忆增长撑爆工具返回上限);
    full=True 返回旧版完整列表(全量,记忆多时可能很大,慎用)。include_archive=True含归档。
    要看某桶原文用 inspect(bucket_id),要精确找用 search(关键词)。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    cw = (config.get("current_world") or "").strip() or "日常模式 (空)"
    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"当前世界: {cw}\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
        f"整理引擎: {'运行中' if consolidation_engine.is_running else '已停止'}\n"
        f"情节引擎: {'运行中' if episode_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    # 先算分排序：默认只列权重最高的 limit 个，封顶工具返回大小，记忆再涨也不撑爆。
    scored = []
    for b in buckets:
        meta = b.get("metadata", {})
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        scored.append((b, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    if full:
        lines = [_format_pulse_line(b, score, full=True) for b, score in scored]
        return status + "\n=== 记忆列表 (全量 {0} 个) ===\n".format(len(scored)) + "\n".join(lines)

    shown = scored if limit <= 0 else scored[:limit]
    lines = [_format_pulse_line(b, score, full=False) for b, score in shown]
    omitted = len(scored) - len(shown)
    footer = ""
    if omitted > 0:
        footer = (
            f"\n…还有 {omitted} 个权重较低的桶未列出。"
            "用 inspect(bucket_id) 看原文、search(关键词) 精确找、pulse(full=True) 拉全量。"
        )
    return (
        status
        + f"\n=== 记忆导航 (Top {len(shown)} / 共 {len(scored)}) ===\n"
        + "默认按权重摘要；完整原文用 inspect(bucket_id)，精确找用 search(关键词)，全量用 pulse(full=True)。\n"
        + "\n".join(lines)
        + footer
    )


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """做梦——读取最近新增的记忆桶,供你自省。读完后可以trace(resolved=1)放下,或hold(feel=True)写感受。"""
    await _ensure_decay_background()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel) ---
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]

    # --- Sort by creation time desc, take top 10 ---
    candidates.sort(
        key=lambda b: event_at_from_metadata(b["metadata"]) or "",
        reverse=True,
    )
    recent = candidates[:10]

    if not recent:
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = event_at_from_metadata(meta) or ""
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint


# =============================================================
# Helper: split recent buckets into "current window" vs "prior windows"
# 辅助函数:把最近活跃桶按时间 gap 拆成「上一窗口」+「再之前」两组
#
# Pure function (no side effects, no async); easy to unit-test.
# 纯函数,易测试。
# =============================================================
def _split_recent_by_time_gap(
    buckets: list,
    gap_threshold_seconds: int = 3600,
    window_cap: int = 5,
    prior_cap: int = 3,
) -> tuple[list, list]:
    """
    Split a list of buckets (already sorted by last_active desc) into:
      - recent_window: buckets from the most recent contiguous time cluster
      - prior_windows: buckets from earlier clusters

    Detection: find the largest gap between consecutive last_active timestamps.
    If max_gap >= gap_threshold_seconds, that gap is the window boundary.
    Otherwise treat the whole list as one continuous window.

    把按 last_active 降序排好的桶拆成两组:
      - recent_window: 上一窗口(最新一段连续时间团)
      - prior_windows: 再之前(更早的时间团)
    用最大时间 gap 检测窗口边界,gap 不够阈值就全归 recent_window。
    """
    def _parse_ts(s: str):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    ts_pairs = [
        (_parse_ts(b["metadata"].get("last_active", "")), b) for b in buckets
    ]
    ts_pairs = [(t, b) for t, b in ts_pairs if t is not None]

    if not ts_pairs:
        return ([], [])
    if len(ts_pairs) == 1:
        return ([ts_pairs[0][1]], [])

    gaps = [
        ((ts_pairs[i][0] - ts_pairs[i + 1][0]).total_seconds(), i)
        for i in range(len(ts_pairs) - 1)
    ]
    max_gap, split_idx = max(gaps, key=lambda x: x[0])

    if max_gap >= gap_threshold_seconds:
        recent_window = [b for _, b in ts_pairs[: split_idx + 1]][:window_cap]
        prior_windows = [b for _, b in ts_pairs[split_idx + 1 :]][:prior_cap]
    else:
        recent_window = [b for _, b in ts_pairs][:window_cap]
        prior_windows = []

    return (recent_window, prior_windows)


def _created_within_days(b: dict, max_age_days: float, now: datetime = None) -> bool:
    """桶的 event_at（事件真正发生时间）是否落在最近 max_age_days 天内。

    用于简报「最近活跃」叙事段的绝对年龄闸：last_active 会被 inspect/
    backfill_relations/touch/update 等维护操作 bump，旧桶会冒充「最近活跃」
    被 LLM 写成「前两天」。改用 event_at 判事件年龄，旧桶踢出叙事段
    （仍可走 pinned/protected/未解决权重池）。

    event_at 缺失或不可解析 → False（保守踢出，宁缺毋滥防旧桶漏网）。
    (2026-06-08 修：朝灯戳穿一个月前的卡兜事被当成前两天)
    """
    raw = event_at_from_metadata(b.get("metadata", {}) or {}) or ""
    if not raw:
        return False
    try:
        ev = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return False
    try:
        # naive/aware 各取对应 now，避免 offset-naive 减 offset-aware 报错。
        ref = (now or (datetime.now(ev.tzinfo) if ev.tzinfo is not None else datetime.now()))
        return (ref - ev).total_seconds() <= max_age_days * 86400
    except Exception:
        return False


def _filter_briefing_currentness(buckets: list[dict]) -> list[dict]:
    """Keep only records safe to present as current fresh-window context.

    Historical experiences remain available for dated recall.  This gate only
    removes explicit stale-state conflicts from the fresh-window pack:
    superseded/non-current facts and unaudited operational status snapshots.
    """
    candidates = list(buckets or [])
    if _operational_status_validity_enabled():
        try:
            candidates = _get_operational_status_validity_store().attach(candidates)
        except Exception as exc:
            logger.warning(
                "Briefing currentness marker lookup failed closed for status facts: %s",
                type(exc).__name__,
            )

    kept: list[dict] = []
    dropped = {
        "superseded": 0,
        "non_current_fact": 0,
        "unaudited_status": 0,
    }
    for bucket in candidates:
        if not isinstance(bucket, dict):
            continue
        meta = bucket.get("metadata", {}) or {}
        if (
            str(meta.get("superseded_by_bucket_id") or "").strip()
            or str(meta.get("validity_superseded_by_bucket_id") or "").strip()
        ):
            dropped["superseded"] += 1
            continue

        if "fact_status" in meta:
            fact_status = str(meta.get("fact_status") or "").strip().lower()
            if fact_status != FACT_STATUS_CURRENT:
                dropped["non_current_fact"] += 1
                continue

        status_label = operational_validity_label(
            bucket,
            view=OPERATIONAL_VIEW_CURRENT,
        )
        if status_label:
            status_state = status_label.get("state")
            if status_state != OPERATIONAL_STATE_CURRENT:
                briefing_cfg = config.get("briefing", {}) or {}
                try:
                    fallback_days = float(
                        briefing_cfg.get("unaudited_status_max_age_days", 1)
                    )
                except (TypeError, ValueError):
                    fallback_days = 1.0
                fresh_unresolved = (
                    status_state == "unknown"
                    and not bool(meta.get("resolved", False))
                    and _created_within_days(
                        bucket,
                        max(0.0, min(fallback_days, 1.0)),
                    )
                )
                if not fresh_unresolved:
                    dropped["unaudited_status"] += 1
                    continue
        kept.append(bucket)

    if any(dropped.values()):
        logger.info(
            "Briefing currentness gate removed stale candidates: "
            "superseded=%d non_current_fact=%d unaudited_status=%d",
            dropped["superseded"],
            dropped["non_current_fact"],
            dropped["unaudited_status"],
        )
    return kept


def _event_age_label(b: dict, now: datetime = None) -> str:
    """简报素材里每个浮现桶的硬日期章——给 LLM 一个不可忽略的绝对时间锚。

    有 event_at → "发生于 2026-05-30（距今 N 天）"；
    event_at 缺失/不可解析 → "⚠ 无确切日期，禁止叙述为「近期/前两天/刚刚」"。

    根治：旧桶从「高权重未解决」「感情红线」等非叙事池冒出来时，如果日期没有
    跟着素材一起进入最终消费路径，LLM 会自行编「前两天」。2026-06-08 的
    _created_within_days 闸只管「最近活跃」池；这里给所有简报路径统一盖章。
    """
    NO_DATE = "⚠ 无确切日期，禁止叙述为「近期/前两天/刚刚/最近」，只作背景、不带时间词"
    raw = event_at_from_metadata(b.get("metadata", {}) or {}) or ""
    if not raw:
        return NO_DATE
    try:
        ev = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return NO_DATE
    try:
        if ev.tzinfo is not None:
            ref = now or datetime.now(ev.tzinfo)
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=ev.tzinfo)
            event_date = ev.astimezone(_BJ_TZ).date()
            ref_date = ref.astimezone(_BJ_TZ).date()
        else:
            # 历史 created 多为无时区字符串；其日期部分是唯一可审计的日历锚，
            # 不擅自平移。当前日期统一取北京时间，避免 NAS 容器 UTC 跨日错标。
            ref = now or datetime.now(_BJ_TZ)
            if ref.tzinfo is not None:
                ref = ref.astimezone(_BJ_TZ).replace(tzinfo=None)
            event_date = ev.date()
            ref_date = ref.date()
        days = (ref_date - event_date).days
    except Exception:
        return NO_DATE
    date_text = event_date.isoformat()
    if days < 0:
        return f"⚠ 日期 {date_text} 晚于当前日期，禁止叙述为已发生事件"
    when = "今天" if days == 0 else ("昨天" if days == 1 else f"距今 {days} 天")
    return f"发生于 {date_text}（{when}）"


def _format_dated_raw_slot_text(b: dict) -> str:
    """JSON 原文 slot 的正文；日期必须进入 consumer 实际读取的 text 字段。"""
    body = redact_text(strip_wikilinks(b.get("content", ""))).strip()
    return f"📅 {_event_age_label(b)}\n{body}"


def _is_protected_domain_bucket(b: dict) -> bool:
    """命中 PROTECTED_RESOLVE_DOMAINS（恋爱/约定/纪念日/家庭/自省/feel）的桶。
    这些是感情红线域，简报里不许被 LLM 压缩成「一行任务」。"""
    doms = set((b.get("metadata", {}) or {}).get("domain", []) or [])
    return bool(doms & PROTECTED_RESOLVE_DOMAINS)


def _format_protected_verbatim(b: dict) -> str:
    """把保护域桶格式化成「逐字不可压缩 + 打标」原文块（绕开 LLM 压缩）。
    根治 #3（开窗只啃脱水简报→把恋爱/约定当 resolved 读）
    + #15（LLM 把活约定/红线 paraphrase 成一行任务）。"""
    meta = b.get("metadata", {}) or {}
    name = meta.get("name", b["id"])
    doms = ",".join(meta.get("domain", []) or [])
    body = redact_text(strip_wikilinks(b.get("content", ""))).strip()
    return (
        f"【原文·{doms}】{name}（id={b['id']}）\n"
        f"📅 {_event_age_label(b)}\n"
        f"{body}\n"
        f"⚠ 此为原文片段、逐字保真、未经压缩。触及须 inspect 桶 id={b['id']} 读全文；"
        f"禁止当 resolved/已完成/演的/迎合 处理。"
    )


def _format_bucket_for_briefing(b: dict, section_tag: str) -> str:
    """
    Format a single bucket as raw material for the briefing LLM.

    Output is a multi-line text block per bucket with:
      - [section_tag] name
      - domain / tags
      - V/A/importance/last_active
      - emotion (if extractable from dehydrated JSON content)
      - optional emotion scaffold lines for non-feel buckets
      - first 400 chars of content (wikilinks stripped)

    The `emotion` line is critical: dehydrated content is JSON-stringified
    `{"core_facts": [...], "emotion_state": "...", ...}`. The first 400 chars
    are often eaten by `core_facts`, truncating `emotion_state` away. Even
    when present, LLMs tend to drop structured labels during compression.
    Extracting it to its own labeled line + the BRIEFING_PROMPT emotion-field
    rule (see dehydrator.py) is double-insurance against emotion erasure.

    把单个桶格式化成简报 LLM 的原始素材。
    emotion 字段独立成行+ prompt 铁律,双保险防止脱水时锁定的情绪关键词被压没。
    非 feel 桶额外暴露情绪脚手架 wire 行,供 prompt 决定靠近方式。
    """
    meta = b["metadata"]
    name = meta.get("name", b["id"])
    doms = ",".join(meta.get("domain", []) or [])
    tags = ",".join((meta.get("tags", []) or [])[:10])
    val = meta.get("valence", 0.5)
    aro = meta.get("arousal", 0.3)
    imp = meta.get("importance", 5)
    last_active = meta.get("last_active", "")
    raw_content = b.get("content", "")

    def _clean_string(value) -> str:
        return value.strip() if isinstance(value, str) else ""

    def _clean_string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    # --- Extract structured emotion fields from dehydrated JSON content ---
    # --- 从脱水 JSON content 抽出结构化情绪字段 ---
    parsed = None
    emotion = ""
    try:
        loaded = json.loads(raw_content) if raw_content else None
        if isinstance(loaded, dict):
            parsed = loaded
            emotion = _clean_string(parsed.get("emotion_state"))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    scaffold_keys = {
        "body_signal",
        "unspoken_need",
        "sore_point",
        "response_rule",
        "do_not",
        "sample_voice",
    }
    body_source = raw_content
    if parsed is not None:
        body_source = json.dumps(
            {k: v for k, v in parsed.items() if k not in scaffold_keys},
            ensure_ascii=False,
        )
    body = strip_wikilinks(body_source)[:400]

    lines = [
        f"[{section_tag}] {name}",
        f"  domain:{doms} | tags:{tags}",
        f"  V{val:.2f}/A{aro:.2f} 重要:{imp} last_active:{last_active}",
        f"  📅 {_event_age_label(b)}",
    ]
    if emotion:
        lines.append(f"  emotion:{emotion}")
    if parsed is not None and meta.get("type") != "feel":
        scaffold = [
            ("body", _clean_string(parsed.get("body_signal"))),
            ("need", _clean_string(parsed.get("unspoken_need"))),
            ("sore", _clean_string(parsed.get("sore_point"))),
            ("approach", _clean_string(parsed.get("response_rule"))),
            ("avoid", " / ".join(_clean_string_list(parsed.get("do_not")))),
            ("voice", " | ".join(_clean_string_list(parsed.get("sample_voice")))),
        ]
        for label, value in scaffold:
            if value:
                lines.append(f"  {label}:{value}")
    lines.append(f"  {body}")
    return "\n".join(lines)


def _anchor_label_for_bucket(bucket: dict, max_chars: int = 48) -> str:
    """短标签只给锚索引用；bucket_id 才是反查主键。"""
    meta = bucket.get("metadata", {}) or {}
    label = str(meta.get("name") or "").strip()
    if not label:
        label = _bucket_navigator_summary(bucket, max_chars=max_chars)
    label = strip_wikilinks(_strip_markdown_images(label))
    label = _collapse_ws(label)
    return _clip_text(label or bucket.get("id", "unknown"), max_chars)


def _anchor_priority(bucket: dict) -> int:
    """锚索引稳定排序：核心/高正向桶靠前，其余保持召回顺序。"""
    meta = bucket.get("metadata", {}) or {}
    domains = [str(x) for x in (meta.get("domain", []) or [])]
    tags = [str(x) for x in (meta.get("tags", []) or [])]
    label = str(meta.get("name") or "")
    haystack = " ".join([label, *domains, *tags])
    try:
        valence = float(meta.get("valence", 0.5) or 0.5)
    except (TypeError, ValueError):
        valence = 0.5
    if meta.get("pinned") or meta.get("protected"):
        return 0
    if valence >= 0.8:
        return 1
    if "核心" in haystack:
        return 2
    return 3


def _format_anchor_index(buckets: list[dict]) -> str:
    """给 briefing/dream prompt 追加 bucket_id 反查表，UI 会在 === 块前截断。"""
    seen: set[str] = set()
    indexed: list[tuple[int, dict]] = []
    for idx, bucket in enumerate(buckets or []):
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id or bucket_id in seen:
            continue
        seen.add(bucket_id)
        indexed.append((idx, bucket))
    if not indexed:
        return ""

    indexed.sort(key=lambda item: (_anchor_priority(item[1]), item[0]))
    lines = ["=== 锚索引 ==="]
    for _, bucket in indexed:
        bucket_id = str(bucket.get("id") or "").strip()
        lines.append(f"src: [{bucket_id}] {_anchor_label_for_bucket(bucket)}")
    return "\n".join(lines)


def _append_anchor_index(text: str, anchor_index: str) -> str:
    anchor_index = (anchor_index or "").strip()
    if not anchor_index:
        return text
    return f"{text.rstrip()}\n\n{anchor_index}"


def _generated_briefing_enabled() -> bool:
    """Whether any automatic briefing path may call the generation model."""
    briefing_cfg = config.get("briefing", {}) or {}
    return bool(briefing_cfg.get("generated_enabled", True))


def _format_deterministic_boot_hooks(
    buckets: list[dict],
    max_chars: int,
) -> str:
    """Build a bounded local hot index without invoking a generation model."""
    budget = max(120, int(max_chars))
    lines = ["以下是本地确定性记忆钩子；细节按 bucket id inspect 原文："]
    seen: set[str] = set()
    for bucket in buckets or []:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id or bucket_id in seen:
            continue
        seen.add(bucket_id)
        meta = bucket.get("metadata", {}) or {}
        label = _anchor_label_for_bucket(bucket, max_chars=42)
        event_at = str(event_at_from_metadata(meta) or "").strip()
        date_label = event_at[:10] if event_at else "无确切日期"
        summary = _bucket_navigator_summary(bucket, max_chars=88)
        line = f"- [{bucket_id}] {label} | {date_label} | {summary}"
        candidate = "\n".join([*lines, line])
        if len(candidate) > budget:
            remaining = budget - len("\n".join(lines)) - 1
            if remaining >= 24:
                lines.append(_clip_text(line, remaining))
            break
        lines.append(line)
    if len(lines) == 1:
        lines.append("- 当前没有可列入开机索引的记忆。")
    return "\n".join(lines)[:budget]


def _render_deterministic_boot_pack(
    *,
    time_header: str,
    protected_verbatim: list[dict],
    tier0_buckets: list[dict],
    selected_buckets: list[dict],
    anchor_index: str,
    max_chars: int,
    format: str,
) -> str:
    """Render a fresh-window pack from stored records only; no LLM path."""
    hooks = _format_deterministic_boot_hooks(selected_buckets, max_chars)
    if format != "json":
        original_parts = [
            _format_protected_verbatim(bucket)
            for bucket in protected_verbatim
        ]
        original_parts.extend(
            _format_dated_raw_slot_text(bucket)
            for bucket in tier0_buckets
        )
        return "\n\n".join(
            part for part in [f"# {time_header}", *original_parts, hooks] if part
        )

    slots: list[dict] = []
    for bucket in protected_verbatim:
        meta = bucket.get("metadata", {}) or {}
        slots.append({
            "tier": 0,
            "protected": True,
            "bucket_id": bucket["id"],
            "label": meta.get("name", bucket["id"]),
            "domain": meta.get("domain", []) or [],
            "event_at": event_at_from_metadata(meta),
            "created": event_at_from_metadata(meta),
            "age_label": _event_age_label(bucket),
            "text": _format_dated_raw_slot_text(bucket),
            "warn": (
                f"原文逐字、未压缩。触及须 inspect 桶 id={bucket['id']}；"
                "禁止当 resolved/已完成/演的 处理。"
            ),
        })
    for bucket in tier0_buckets:
        meta = bucket.get("metadata", {}) or {}
        slots.append({
            "tier": 0,
            "label": meta.get("name", bucket["id"]),
            "bucket_id": bucket["id"],
            "event_at": event_at_from_metadata(meta),
            "created": event_at_from_metadata(meta),
            "age_label": _event_age_label(bucket),
            "text": _format_dated_raw_slot_text(bucket),
        })
    slots.append({
        "tier": 1,
        "source": "deterministic",
        "label": "当前事实与未完成事项",
        "text": hooks,
    })
    return json.dumps(
        {
            "time_header": time_header,
            "mode": "deterministic",
            "model_called": False,
            "slots": slots,
            "briefing": hooks,
            "anchor_index": anchor_index,
        },
        ensure_ascii=False,
    )


# =============================================================
# Feel surfacing pool — feel buckets don't decay (score ~50), so they're
# picked by pinned → importance → recency instead of weight. Shared by
# briefing / breath / breath_hook so 情感沉淀 surfaces automatically,
# not only via the on-demand domain="feel" channel.
# feel 不衰减(score 恒 50)，按 pinned→重要度→最近活跃选 top N，三处浮现共用。
# =============================================================
FEEL_SURFACE_CAP = 3


def _surface_feel_pool(all_buckets: list, seen_ids: set = None, cap: int = FEEL_SURFACE_CAP) -> list:
    seen = seen_ids or set()
    feels = [
        b for b in all_buckets
        if b["metadata"].get("type") == "feel" and b["id"] not in seen
    ]
    feels.sort(
        key=lambda b: (
            1 if b["metadata"].get("pinned") else 0,
            int(b["metadata"].get("importance", 5) or 5),
            str(b["metadata"].get("last_active", "")),
        ),
        reverse=True,
    )
    return feels[:cap]


# =============================================================
# Briefing pre-generation cache
# 简报后台预生成缓存
# =============================================================
def _briefing_env_seconds(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


BRIEFING_REFRESH_INTERVAL_SECONDS = _briefing_env_seconds(
    "OMBRE_BRIEFING_REFRESH_INTERVAL_SECONDS",
    3600.0,
    30.0,
)
BRIEFING_REFRESH_TIMEOUT_SECONDS = _briefing_env_seconds(
    "OMBRE_BRIEFING_REFRESH_TIMEOUT_SECONDS",
    120.0,
    15.0,
)

# A profile is exactly the request shape that affects briefing material or
# representation.  Body-state rendering remains request-time work and is not
# cached; session_id stays in the key so request-time state remains isolated,
# while profiles with the same LLM generation shape may reuse immutable
# briefing material instead of regenerating it for every new window.
_BriefingProfile = tuple[int, str, bool, str, str, str]
_briefing_cache_lock = threading.Lock()
_briefing_prebuilt_cache: dict[_BriefingProfile, dict] = {}
_briefing_profiles: set[_BriefingProfile] = {
    (1000, "", False, "", "text", ""),
    (1500, "", False, "", "json", ""),
}
_briefing_refresh_event: asyncio.Event | None = None
_briefing_refresh_task: asyncio.Task | None = None
_briefing_refresh_loop: asyncio.AbstractEventLoop | None = None
_briefing_background_refresh = contextvars.ContextVar(
    "briefing_background_refresh",
    default=False,
)
_briefing_timeout_override = contextvars.ContextVar(
    "briefing_timeout_override",
    default=None,
)


def _briefing_profile(
    max_chars: int,
    domain: str,
    pinned_only: bool,
    session_id: str,
    format: str,
) -> _BriefingProfile:
    return (
        max(300, min(int(max_chars), 4000)),
        (domain or "").strip(),
        bool(pinned_only),
        (session_id or "").strip(),
        "json" if format == "json" else "text",
        (config.get("current_world", "") or "").strip(),
    )


def _briefing_generation_shape(profile: _BriefingProfile) -> tuple:
    """Fields that affect LLM material; session only affects request rendering."""
    return (profile[0], profile[1], profile[2], profile[4], profile[5])


def _register_briefing_profile(profile: _BriefingProfile) -> None:
    shape = _briefing_generation_shape(profile)
    with _briefing_cache_lock:
        _briefing_profiles.add(profile)
        if profile in _briefing_prebuilt_cache:
            return
        shared_entry = next(
            (
                entry
                for cached_profile, entry in _briefing_prebuilt_cache.items()
                if _briefing_generation_shape(cached_profile) == shape
            ),
            None,
        )
        if shared_entry is not None:
            _briefing_prebuilt_cache[profile] = dict(shared_entry)


def _get_briefing_cache_entry(profile: _BriefingProfile) -> dict | None:
    with _briefing_cache_lock:
        return _briefing_prebuilt_cache.get(profile)


def _store_briefing_cache_entry(
    profile: _BriefingProfile,
    *,
    text: str,
    time_header: str,
    buckets: list[dict],
) -> None:
    entry = {
        "text": text,
        "time_header": time_header,
        "format": profile[4],
        "buckets": list(buckets),
        "generated_at": time.time(),
    }
    shape = _briefing_generation_shape(profile)
    with _briefing_cache_lock:
        for registered in _briefing_profiles:
            if _briefing_generation_shape(registered) == shape:
                _briefing_prebuilt_cache[registered] = dict(entry)
        _briefing_prebuilt_cache[profile] = entry
    logger.info(
        "Briefing prebuilt cache refreshed max_chars=%d format=%s session=%s",
        profile[0],
        profile[4],
        "set" if profile[3] else "empty",
    )


def _render_briefing_cache_entry(
    entry: dict,
    *,
    session_id: str,
    include_body_state: bool,
    reset_body_state: bool,
) -> str:
    """Render a cached snapshot with a request-time clock and body state."""
    fresh_header = _now_bj_header()
    cached_text = str(entry.get("text") or "")
    if entry.get("format") == "json":
        payload = json.loads(cached_text)
        payload["time_header"] = fresh_header
        return json.dumps(payload, ensure_ascii=False)

    old_header = str(entry.get("time_header") or "")
    old_prefix = f"# {old_header}\n" if old_header else ""
    if old_prefix and cached_text.startswith(old_prefix):
        cached_text = f"# {fresh_header}\n" + cached_text[len(old_prefix):]
    return _append_body_state_block(
        cached_text,
        list(entry.get("buckets") or []),
        session_id,
        include_body_state,
        reset_body_state,
    )


def _mark_briefing_cache_dirty(reason: str) -> None:
    """Wake the background loop after a durable hold/grow write."""
    event = _briefing_refresh_event
    loop = _briefing_refresh_loop
    if event is None or loop is None or loop.is_closed():
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        event.set()
    else:
        loop.call_soon_threadsafe(event.set)
    logger.debug("Briefing prebuilt cache marked dirty: %s", reason)


async def _refresh_briefing_profile(profile: _BriefingProfile) -> None:
    before = _get_briefing_cache_entry(profile)
    background_token = _briefing_background_refresh.set(True)
    timeout_token = _briefing_timeout_override.set(
        BRIEFING_REFRESH_TIMEOUT_SECONDS
    )
    try:
        await briefing(
            max_chars=profile[0],
            domain=profile[1],
            pinned_only=profile[2],
            session_id=profile[3],
            include_body_state=False,
            reset_body_state=False,
            format=profile[4],
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Briefing background refresh failed; preserving previous cache "
            "max_chars=%d format=%s",
            profile[0],
            profile[4],
        )
    finally:
        _briefing_timeout_override.reset(timeout_token)
        _briefing_background_refresh.reset(background_token)
    after = _get_briefing_cache_entry(profile)
    if after is before:
        logger.error(
            "Briefing background refresh produced no usable summary; "
            "preserving previous cache max_chars=%d format=%s",
            profile[0],
            profile[4],
        )


async def _briefing_cache_refresh_worker() -> None:
    event = _briefing_refresh_event
    if event is None:
        return
    loop = asyncio.get_running_loop()
    last_refresh_completed_at: float | None = None
    logger.info(
        "Briefing background pre-generation started minimum_gap=%.1fs timeout=%.1fs",
        BRIEFING_REFRESH_INTERVAL_SECONDS,
        BRIEFING_REFRESH_TIMEOUT_SECONDS,
    )
    while True:
        # Refresh only when startup or a durable write marks the cache dirty.
        # The old timeout loop regenerated every known profile while idle and
        # could immediately run again when writes arrived during a refresh.
        # A minimum gap keeps those writes coalesced into one bounded batch.
        await event.wait()
        event.clear()
        if last_refresh_completed_at is not None:
            remaining = BRIEFING_REFRESH_INTERVAL_SECONDS - (
                loop.time() - last_refresh_completed_at
            )
            if remaining > 0:
                logger.info(
                    "Briefing background refresh coalescing dirty writes for %.1fs",
                    remaining,
                )
                await asyncio.sleep(remaining)
                # Every dirty mark received during the gap belongs to this
                # refresh batch.  Marks raised while refreshing remain set and
                # schedule one later batch instead of being lost.
                event.clear()
        with _briefing_cache_lock:
            # session_id is a cache-key boundary, not an LLM-input boundary.
            # Refresh one representative per material shape, then _store...
            # fans the result out to every registered session key.
            by_shape: dict[tuple, _BriefingProfile] = {}
            current_world = (config.get("current_world", "") or "").strip()
            for profile in sorted(
                _briefing_profiles,
                key=lambda item: (
                    0
                    if item[0] == 1500 and item[4] == "json" and not item[3]
                    else 1,
                    item,
                ),
            ):
                if profile[5] != current_world:
                    continue
                by_shape.setdefault(_briefing_generation_shape(profile), profile)
            profiles = list(by_shape.values())
        for profile in profiles:
            await _refresh_briefing_profile(profile)
        last_refresh_completed_at = loop.time()


async def _start_briefing_cache_refresh() -> None:
    global _briefing_refresh_event, _briefing_refresh_task, _briefing_refresh_loop
    if _briefing_refresh_task is not None and not _briefing_refresh_task.done():
        return
    if not _generated_briefing_enabled():
        logger.info("Generated briefing disabled; background refresh worker not started")
        return
    current_world = (config.get("current_world", "") or "").strip()
    _register_briefing_profile((1000, "", False, "", "text", current_world))
    _register_briefing_profile((1500, "", False, "", "json", current_world))
    _briefing_refresh_loop = asyncio.get_running_loop()
    _briefing_refresh_event = asyncio.Event()
    _briefing_refresh_event.set()
    _briefing_refresh_task = asyncio.create_task(
        _briefing_cache_refresh_worker(),
        name="briefing-prebuild",
    )


async def _stop_briefing_cache_refresh() -> None:
    global _briefing_refresh_event, _briefing_refresh_task, _briefing_refresh_loop
    task = _briefing_refresh_task
    _briefing_refresh_task = None
    _briefing_refresh_event = None
    _briefing_refresh_loop = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# =============================================================
# Tool 7: briefing — Open-window handoff briefing
# 工具 7：briefing — 开窗交接简报
#
# Aggregates pinned + top-weighted unresolved + recently-active
# buckets, compresses via LLM into a ≤1500-char briefing.
# 聚合钉选 + 高权重未解决 + 最近活跃桶，LLM 压缩为 ≤1500 字简报。
# Designed to replace the 18000-token full-breath open-window cost
# with a 3000-token briefing (~80% savings).
# 用于替代开窗时 18000 token 的完整 breath 浮现，压到约 3000 token。
# =============================================================
@mcp.tool()
async def briefing(
    max_chars: int = 1000,
    domain: str = "",
    pinned_only: bool = False,
    session_id: str = "",
    include_body_state: bool = True,
    reset_body_state: bool = False,
    format: str = "text",
    deterministic: bool = False,
) -> str:
    """开窗记忆包。deterministic=True 时只做本地选择与钩子拼装，绝不调用 LLM。"""
    await _ensure_decay_background()
    max_chars = max(300, min(max_chars, 4000))
    format = "json" if format == "json" else "text"
    if not _generated_briefing_enabled():
        deterministic = True
    profile = _briefing_profile(
        max_chars,
        domain,
        pinned_only,
        session_id,
        format,
    )
    if not deterministic:
        _register_briefing_profile(profile)
    is_background_refresh = _briefing_background_refresh.get()
    if not deterministic and not is_background_refresh and not reset_body_state:
        cached = _get_briefing_cache_entry(profile)
        if cached is not None:
            try:
                return _render_briefing_cache_entry(
                    cached,
                    session_id=session_id,
                    include_body_state=include_body_state,
                    reset_body_state=False,
                )
            except Exception:
                logger.exception(
                    "Briefing prebuilt cache render failed; falling back to cold path"
                )
        _mark_briefing_cache_dirty("request_cache_miss")

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Briefing failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Domain filter ---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()]
    if domain_filter:
        def _domain_hit(b):
            doms = b["metadata"].get("domain", []) or []
            return any(d in doms for d in domain_filter)
        all_buckets = [b for b in all_buckets if _domain_hit(b)]

    # --- World filter (#5 串台修复 2026-05-30)：排除非当前世界线的角色扮演桶 ---
    # feel 桶跨世界、保留；其余按 current_world 过滤（world="" 日常 + "通用" 总通过）。
    # 复用 search 同款 _resolve_world_filter，避免谢长夜/宁亲王世界线渗进日常简报。
    _wf_set = set(_resolve_world_filter("", config.get("current_world", "")))
    all_buckets = [
        b for b in all_buckets
        if b["metadata"].get("type") == "feel"
        or world_matches(b["metadata"].get("world", ""), _wf_set)
    ]

    # A fresh-window pack speaks in the present tense.  Explicitly old fact or
    # status records stay searchable, but cannot masquerade as current context.
    all_buckets = _filter_briefing_currentness(all_buckets)

    # --- Pinned/protected: always included as core principles ---
    # --- 钉选/protected:必入,作为核心准则 ---
    pinned = [
        b for b in all_buckets
        if b["metadata"].get("pinned") or b["metadata"].get("protected")
    ]

    # --- Unresolved by weight (top 10), excluding pinned/feel ---
    # --- 未解决按权重 top 10,排除 pinned/feel ---
    unresolved_pool = [
        b for b in all_buckets
        if not b["metadata"].get("resolved", False)
        and b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
    ]
    unresolved_pool.sort(
        key=lambda b: decay_engine.calculate_score(b["metadata"]),
        reverse=True,
    )
    top_unresolved = unresolved_pool[:10] if not pinned_only else []

    # --- Recently active: split by largest time-gap into current vs prior windows ---
    # --- 最近活跃:按最大时间 gap 拆成「上一窗口」(主体情绪源)+「再之前」(过渡背景) ---
    # 解决"今早吵架桶和昨晚和弦桶一起进、权重相同"的问题：让 LLM 看到时间梯度，
    # 末尾「现在的体感」取自上一窗口而非更早窗口的紧绷。
    seen_ids = {b["id"] for b in pinned} | {b["id"] for b in top_unresolved}
    recent_pool = [
        b for b in all_buckets
        if b["id"] not in seen_ids
        and b["metadata"].get("type") not in ("feel",)
    ]
    # --- 绝对年龄闸（2026-06-08 修）---
    # last_active 会被 inspect/backfill_relations/touch/update 等维护操作 bump，
    # 导致一个月前的桶冒充「最近活跃」、被 LLM 写成「前两天 XXX」。
    # 改用 created（事件真正发生时间）做硬性上限：created 早于窗口的桶踢出最近活跃叙事，
    # 它们若仍重要，自会走 pinned/protected/未解决权重池——不许混进「上一窗口/再之前」。
    _recent_max_age_days = (config.get("briefing", {}) or {}).get("recent_max_age_days", 7)
    recent_pool = [
        b for b in recent_pool
        if _created_within_days(b, _recent_max_age_days)
    ]
    recent_pool.sort(
        key=lambda b: b["metadata"].get("last_active", ""),
        reverse=True,
    )
    recent_window: list = []
    prior_windows: list = []
    if not pinned_only:
        recent_window, prior_windows = _split_recent_by_time_gap(recent_pool[:10])

    # --- Feel buckets: surface as emotional sediment (independent pool) ---
    # --- feel 桶:作为情感沉淀独立浮现(不衰减,与 pinned 去重)---
    feel_seen = {b["id"] for b in pinned}
    top_feel = _surface_feel_pool(all_buckets, feel_seen) if not pinned_only else []

    time_header = _now_bj_header()

    briefing_buckets = pinned + top_unresolved + recent_window + prior_windows + top_feel
    selected_for_boot = list(briefing_buckets)
    anchor_index = _format_anchor_index(briefing_buckets)

    # --- #3+#15 感情红线逐字保真（2026-05-30）---
    # 命中 PROTECTED_RESOLVE_DOMAINS（恋爱/约定/纪念日/家庭/自省/feel）的桶，抽出来、
    # 绕开 LLM 压缩、原文+打标带出——根治"开窗只啃脱水简报→把恋爱/约定当 resolved 读"。
    # 按 last_active 取最近 N 条（默认 6，config.briefing.protected_verbatim_limit 可调），
    # 防止全量原文撑爆简报；未入选的保护域桶仍走压缩 pool。
    _pv_limit = (config.get("briefing", {}) or {}).get("protected_verbatim_limit", 6)
    _protected_pool = [b for b in briefing_buckets if _is_protected_domain_bucket(b)]
    _protected_pool.sort(key=lambda b: b["metadata"].get("last_active", ""), reverse=True)
    protected_verbatim = [] if pinned_only else _protected_pool[:_pv_limit]
    protected_ids = {b["id"] for b in protected_verbatim}
    # 从各压缩 pool 移除——这些桶不进 LLM raw_material
    pinned = [b for b in pinned if b["id"] not in protected_ids]
    top_unresolved = [b for b in top_unresolved if b["id"] not in protected_ids]
    recent_window = [b for b in recent_window if b["id"] not in protected_ids]
    prior_windows = [b for b in prior_windows if b["id"] not in protected_ids]
    top_feel = [b for b in top_feel if b["id"] not in protected_ids]
    protected_block = (
        "## 感情红线·原文逐字区（不可压缩 / 触及须 inspect）\n\n"
        + "\n\n".join(_format_protected_verbatim(b) for b in protected_verbatim)
    ) if protected_verbatim else ""

    # --- #4 核心画像分离（2026-05-30）：tier==0 的桶单独原文 slots，不进 LLM 压缩 ---
    # format=json 时：把 tier==0 桶从各槽剔出来 → tier0_buckets（原文 slots[]）
    # 剩余桶继续走 sections + dehydrator → briefing 字段
    # format=text 时：旧行为保持不变，tier==0 桶继续走压缩，避免破坏既有 caller
    tier0_buckets: list = []
    if format == "json":
        def _is_tier0(b):
            # protected_verbatim 已单独原文输出，不再重复进 tier0
            return b["metadata"].get("tier") == 0 and b["id"] not in protected_ids
        tier0_buckets = [b for b in briefing_buckets if _is_tier0(b)]
        pinned = [b for b in pinned if not _is_tier0(b)]
        top_unresolved = [b for b in top_unresolved if not _is_tier0(b)]
        recent_window = [b for b in recent_window if not _is_tier0(b)]
        prior_windows = [b for b in prior_windows if not _is_tier0(b)]
        top_feel = [b for b in top_feel if not _is_tier0(b)]
        briefing_buckets = pinned + top_unresolved + recent_window + prior_windows + top_feel

    if deterministic:
        return _render_deterministic_boot_pack(
            time_header=time_header,
            protected_verbatim=protected_verbatim,
            tier0_buckets=tier0_buckets,
            selected_buckets=selected_for_boot,
            anchor_index=anchor_index,
            max_chars=max_chars,
            format=format,
        )

    if not briefing_buckets:
        # 动态素材为空：不调 LLM，直接输出原文级内容（感情红线 protected + tier0 核心画像）
        if format == "json":
            import json as _json
            slots = []
            for b in protected_verbatim:
                meta = b.get("metadata", {})
                slots.append({
                    "tier": 0,
                    "protected": True,
                    "bucket_id": b["id"],
                    "label": meta.get("name", b["id"]),
                    "domain": meta.get("domain", []) or [],
                    "event_at": event_at_from_metadata(meta),
                    "created": event_at_from_metadata(meta),
                    "age_label": _event_age_label(b),
                    "text": _format_dated_raw_slot_text(b),
                    "warn": (
                        f"原文逐字、未压缩。触及须 inspect 桶 id={b['id']}；"
                        f"禁止当 resolved/已完成/演的 处理。"
                    ),
                })
            for b in tier0_buckets:
                meta = b.get("metadata", {})
                slots.append({
                    "tier": 0,
                    "label": meta.get("name", b["id"]),
                    "bucket_id": b["id"],
                    "event_at": event_at_from_metadata(meta),
                    "created": event_at_from_metadata(meta),
                    "age_label": _event_age_label(b),
                    "text": _format_dated_raw_slot_text(b),
                })
            if anchor_index:
                slots.append({
                    "tier": 1,
                    "label": "锚索引",
                    "text": anchor_index,
                })
            output = _json.dumps(
                {
                    "time_header": time_header,
                    "slots": slots,
                    "briefing": anchor_index,
                    "anchor_index": anchor_index,
                },
                ensure_ascii=False,
            )
            _store_briefing_cache_entry(
                profile,
                text=output,
                time_header=time_header,
                buckets=[],
            )
            return output
        _empty_body = (
            f"# {time_header}\n\n{protected_block}" if protected_block
            else f"# {time_header}\n\n记忆库当前空闲，没有可简报的素材。"
        )
        _empty_body = _append_anchor_index(_empty_body, anchor_index)
        _store_briefing_cache_entry(
            profile,
            text=_empty_body,
            time_header=time_header,
            buckets=[],
        )
        return _append_body_state_block(
            _empty_body, [], session_id, include_body_state, reset_body_state
        )

    # --- Build raw material: name + meta + truncated content per bucket ---
    # --- 拼接原始素材:每桶 name + meta + 截断 content ---
    # NOTE: actual formatter is module-level `_format_bucket_for_briefing` (testable).
    # 实现提到模块层面方便测试,这里只是别名。
    _format_bucket = _format_bucket_for_briefing

    sections = []
    if pinned:
        sections.append(
            "=== 核心准则 (pinned) ===\n"
            + "\n\n".join(_format_bucket(b, "pinned") for b in pinned)
        )
    if top_unresolved:
        sections.append(
            "=== 高权重未解决 ===\n"
            + "\n\n".join(_format_bucket(b, "unresolved") for b in top_unresolved)
        )
    if recent_window:
        sections.append(
            "=== 上一窗口 (主体情绪源) ===\n"
            + "\n\n".join(_format_bucket(b, "recent_window") for b in recent_window)
        )
    if prior_windows:
        sections.append(
            "=== 再之前 (过渡背景) ===\n"
            + "\n\n".join(_format_bucket(b, "prior_window") for b in prior_windows)
        )
    if top_feel:
        sections.append(
            "=== 情感沉淀 (feel) ===\n"
            + "\n\n".join(_format_bucket(b, "feel") for b in top_feel)
        )

    # Prepend time header to raw material so the LLM sees the actual time
    # (in case it accidentally reasons about location/weekday despite the rule).
    raw_material = f"=== 当前时点 ===\n{time_header}\n\n" + "\n\n".join(sections)

    # --- Compress via LLM ---
    # 冷启动缺缓存时仍走 Dehydrator 的 15s 总闸；后台预生成通过 context override
    # 放宽到 120s。后台慢慢生成，读路径只消费完整成功产物。
    try:
        timeout_override = _briefing_timeout_override.get()
        if timeout_override is None:
            # Keep compatibility with local test doubles and the established
            # request-time Dehydrator contract.
            result = await dehydrator.briefing(
                raw_material,
                max_chars=max_chars,
            )
        else:
            result = await dehydrator.briefing(
                raw_material,
                max_chars=max_chars,
                total_timeout_seconds=timeout_override,
            )
    except Exception as e:
        logger.error(f"Briefing compression failed: {e}")
        return f"# {time_header}\n\n简报生成失败：{e}"

    if not result:
        return f"# {time_header}\n\n简报生成为空，请稍后重试。"

    result_is_compressed = "简报压缩未完成" not in result
    result_with_anchor = _append_anchor_index(result, anchor_index)

    # --- Stats footer for visibility ---
    stats = (
        f"\n\n---\n"
        f"_素材:{len(pinned)}钉选 / {len(top_unresolved)}未解决 / "
        f"{len(recent_window)}上一窗口 / {len(prior_windows)}再之前 / "
        f"{len(top_feel)}情感沉淀 "
        f"→ 简报{len(result)}字 (~{count_tokens_approx(result)}token)_"
    )

    # Always prepend the real-time header — never trust the LLM to write the date.
    # 永远强制前置时点行——LLM 写不写都不依赖。
    # 感情红线原文区前置——开窗第一眼是逐字原文+打标，而非 LLM 脱水摘要
    _pblock = f"{protected_block}\n\n---\n\n" if protected_block else ""
    text = f"# {time_header}\n\n{_pblock}{result_with_anchor}{stats}"

    # --- #4 format=json 路径：返回 slots[]（每 slot 自带 tier）---
    # tier=0 → 核心画像原文（一桶一 slot）
    # tier=1 → 动态记忆简报（LLM 压缩后整段一个 slot，匹配 claude-twin 消费侧约定）
    # 保留 briefing 字段方便诊断/直接读，消费侧实际按 slots[].tier 分流。
    if format == "json":
        import json as _json
        slots = []
        # 感情红线原文 slots（最高优先，逐字未压缩，带 inspect 警示）
        for b in protected_verbatim:
            meta = b.get("metadata", {})
            slots.append({
                "tier": 0,
                "protected": True,
                "bucket_id": b["id"],
                "label": meta.get("name", b["id"]),
                "domain": meta.get("domain", []) or [],
                "event_at": event_at_from_metadata(meta),
                "created": event_at_from_metadata(meta),
                "age_label": _event_age_label(b),
                "text": _format_dated_raw_slot_text(b),
                "warn": (
                    f"原文逐字、未压缩。触及须 inspect 桶 id={b['id']}；"
                    f"禁止当 resolved/已完成/演的 处理。"
                ),
            })
        for b in tier0_buckets:
            meta = b.get("metadata", {})
            slots.append({
                "tier": 0,
                "label": meta.get("name", b["id"]),
                "bucket_id": b["id"],
                "event_at": event_at_from_metadata(meta),
                "created": event_at_from_metadata(meta),
                "age_label": _event_age_label(b),
                "text": _format_dated_raw_slot_text(b),
            })
        if result:
            slots.append({
                "tier": 1,
                "label": "动态记忆简报",
                "text": result_with_anchor,
            })
        output = _json.dumps(
            {
                "time_header": time_header,
                "slots": slots,
                "briefing": result_with_anchor,
                "anchor_index": anchor_index,
                "stats": stats.strip(),
            },
            ensure_ascii=False,
        )
        if result_is_compressed:
            _store_briefing_cache_entry(
                profile,
                text=output,
                time_header=time_header,
                buckets=briefing_buckets,
            )
        return output

    if result_is_compressed:
        _store_briefing_cache_entry(
            profile,
            text=text,
            time_header=time_header,
            buckets=briefing_buckets,
        )
    return _append_body_state_block(
        text, briefing_buckets, session_id, include_body_state, reset_body_state
    )


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency)."""
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True, include_nsfw=True)  # dashboard 管理：看全部(含涩涩)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
                "event_at": event_at_from_metadata(meta) or "",
                "recorded_at": meta.get("recorded_at", ""),
                "date_precision": meta.get("date_precision", "unknown"),
                "date_source": meta.get("date_source", ""),
                "date_confidence": meta.get("date_confidence"),
                "created": event_at_from_metadata(meta) or "",
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "content_preview": redact_text(strip_wikilinks(b.get("content", "")))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": redact_text(strip_wikilinks(bucket.get("content", ""))),
        "score": decay_engine.calculate_score(meta),
    })

@mcp.custom_route("/api/bucket/{bucket_id}", methods=["POST"])
async def api_bucket_update(request):
    """Update bucket via dashboard. Accepts JSON body with optional fields:
    name, domain (list or csv), tags (list or csv), valence, arousal,
    importance, resolved, pinned, digested, content, image_base64, image_filename.
    Mirrors trace() tool but exposed as HTTP for dashboard editing.
    """
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)

    updates = {}

    if "name" in data and data["name"]:
        updates["name"] = str(data["name"]).strip()

    if "domain" in data and data["domain"]:
        d = data["domain"]
        if isinstance(d, list):
            updates["domain"] = [str(x).strip() for x in d if str(x).strip()]
        else:
            updates["domain"] = [x.strip() for x in str(d).split(",") if x.strip()]

    if "tags" in data and data["tags"]:
        t = data["tags"]
        if isinstance(t, list):
            updates["tags"] = [str(x).strip() for x in t if str(x).strip()]
        else:
            updates["tags"] = [x.strip() for x in str(t).split(",") if x.strip()]

    if "valence" in data:
        try:
            v = float(data["valence"])
            if 0 <= v <= 1:
                updates["valence"] = v
        except (TypeError, ValueError):
            pass

    if "arousal" in data:
        try:
            a = float(data["arousal"])
            if 0 <= a <= 1:
                updates["arousal"] = a
        except (TypeError, ValueError):
            pass

    if "importance" in data:
        try:
            imp = int(data["importance"])
            if 1 <= imp <= 10:
                updates["importance"] = imp
        except (TypeError, ValueError):
            pass

    if "resolved" in data and data["resolved"] is not None:
        updates["resolved"] = bool(data["resolved"])

    if "pinned" in data and data["pinned"] is not None:
        updates["pinned"] = bool(data["pinned"])
        if updates["pinned"]:
            updates["importance"] = 10

    if "digested" in data and data["digested"] is not None:
        updates["digested"] = bool(data["digested"])

    # Content + optional image upload
    new_content = data.get("content", "")
    image_b64 = data.get("image_base64", "")
    image_filename = data.get("image_filename", "image")

    if image_b64 and image_b64.strip():
        try:
            image_url = r2_storage.upload_base64(image_b64, image_filename) or ""
        except Exception as e:
            logger.warning(f"R2 image upload raised in dashboard edit: {e}")
            image_url = ""
        if image_url:
            prefix = f"![{image_filename}]({image_url})\n\n"
            new_content = prefix + (new_content or bucket.get("content", ""))

    if new_content:
        updates["content"] = new_content

    # --- Relation edits（镜像 trace 工具，补齐 #2 去重需要的建边能力）---
    # add_relation: "type:target_id" 或 "type:target_id:note"；remove_relation: "type:target_id" 或 "target_id"
    relation_msgs = []
    _add = str(data.get("add_relation", "") or "").strip()
    if _add:
        parts = [p.strip() for p in _add.split(":", 2)]
        if len(parts) < 2:
            return JSONResponse(
                {"error": "add_relation 格式错误，需 'type:target_id' 或 'type:target_id:note'"},
                status_code=400,
            )
        rel_type, target_id = parts[0], parts[1]
        note = parts[2] if len(parts) >= 3 else ""
        ok = await bucket_mgr.add_relation(bucket_id, target_id, rel_type, note)
        relation_msgs.append({"op": "add", "type": rel_type, "target": target_id, "ok": bool(ok)})
    _rm = str(data.get("remove_relation", "") or "").strip()
    if _rm:
        parts = [p.strip() for p in _rm.split(":", 1)]
        if len(parts) == 1:
            rel_type, target_id = "", parts[0]
        else:
            rel_type, target_id = parts[0], parts[1]
        n = await bucket_mgr.remove_relation(bucket_id, target_id, rel_type)
        relation_msgs.append({"op": "remove", "target": target_id, "removed": n})

    if not updates and not relation_msgs:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    if updates:
        async with bucket_mgr._maintenance_barrier.shared_async():
            success = await bucket_mgr.update(bucket_id, **updates)
            if not success:
                return JSONResponse({"error": "update failed"}, status_code=500)

            if "content" in updates:
                try:
                    await embedding_engine.generate_and_store(bucket_id, updates["content"])
                except Exception:
                    pass

        if success and "content" in updates:
            await _synchronize_bucket_entities(bucket_id, updates["content"])

    return JSONResponse({
        "ok": True,
        "updated": list(updates.keys()),
        "relations": relation_msgs,
    })
@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """Search buckets by query."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await bucket_mgr.search(
            query,
            limit=10,
            relevance_first=True,
        )
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "content_preview": redact_text(strip_wikilinks(b.get("content", "")))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get memory network for visualization.
    Edges have two flavors:
      - kind=relation: explicit 6-type semantic edges from frontmatter `relations`
      - kind=similarity: embedding cosine similarity > 0.5 (background layer)
    """
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}
        bucket_ids = set()

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            bucket_ids.add(bid)
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "digested": meta.get("digested", False),
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Explicit semantic relations (6 types: causes/contributes/improves/explains/updates/kin)
        for b in all_buckets:
            src = b["id"]
            for r in (b.get("metadata", {}).get("relations") or []):
                if not isinstance(r, dict):
                    continue
                tgt = r.get("target")
                rtype = r.get("type")
                if not tgt or not rtype or tgt not in bucket_ids:
                    continue
                edges.append({
                    "source": src,
                    "target": tgt,
                    "kind": "relation",
                    "type": rtype,
                    "note": r.get("note", ""),
                })

        # Embedding similarity edges (background layer, undirected)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({
                        "source": id_a,
                        "target": id_b,
                        "kind": "similarity",
                        "similarity": round(sim, 3),
                    })

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
        }
        secondary_weight = w["emotion"] + w["time"] + w["importance"]

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                secondary_total = (
                    emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                )
                normalized = topic * 100.0
                tie_break_score = (
                    secondary_total / secondary_weight * 100.0
                    if secondary_weight > 0 else 0.0
                )
                resolved = meta.get("resolved", False)
                if resolved:
                    tie_break_score *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                    },
                    "weights": w,
                    "secondary_tie_break": round(tie_break_score, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": (
                        normalized >= bucket_mgr.literal_candidate_floor
                    ),
                })
            except Exception:
                continue

        results = rank_within_relevance_bands(
            results,
            relevance_score=lambda row: row["normalized"],
            tie_break_score=lambda row: row["secondary_tie_break"],
            band_width=bucket_mgr.keyword_relevance_tie_band,
        )
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.literal_candidate_floor,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


OMBRE_RECALL_TRACE_MAX_BYTES = 2_000_000


def _append_recall_status_trace(record: dict) -> None:
    """Content-free, bounded P0 recall trace outside the bucket vault."""
    path = os.environ.get("OMBRE_RECALL_TRACE_PATH", "").strip()
    if not path:
        path = os.path.join(os.path.dirname(config["buckets_dir"]), "recall_status_trace.jsonl")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) >= OMBRE_RECALL_TRACE_MAX_BYTES:
            os.unlink(path)
        payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.warning("recall status trace append skipped: %s", type(exc).__name__)


async def _probe_anchor_status(query: str) -> dict:
    """Run the read-only recall candidate path without touching bucket access time."""
    started = time.monotonic()
    recall_limit = BREATH_RECALL_POOL_SIZE
    recall_query, raw_entity_ranked = _resolve_entity_recall(query)
    intent_policy = _resolve_recall_policy(
        recall_query,
        base_recall_limit=recall_limit,
        requested_relation_depth=0,
    )
    # Hot-path arbitration is deterministic: do not call the generative query
    # expander before every Claude turn. Normal breath keeps its existing
    # expansion behavior; P0 trace makes this boundary explicit.
    query_angles = [recall_query]
    qe_cfg = config.get("query_expansion", {}) or {}

    world_filter = _resolve_world_filter("", config.get("current_world", ""))
    wf_set = {str(value).strip() for value in world_filter} if world_filter is not None else None
    keyword_by_id: dict[str, dict] = {}
    keyword_error = False
    try:
        keyword_batches = await asyncio.gather(*(
            bucket_mgr.search(
                angle,
                limit=intent_policy["keyword_top_k"],
                world_filter=world_filter,
                relevance_first=True,
            )
            for angle in query_angles
        ))
        for batch in keyword_batches:
            for bucket in batch:
                if not _is_main_recall_bucket(bucket):
                    continue
                existing = keyword_by_id.get(bucket["id"])
                if existing is None or bucket.get("score", 0) > existing.get("score", 0):
                    keyword_by_id[bucket["id"]] = bucket
        keyword_by_id = {
            bucket["id"]: bucket
            for bucket in _filter_z_fact_candidates(
                keyword_by_id.values(),
                query=recall_query,
                intent=intent_policy["intent"],
            )
        }
    except Exception as exc:
        logger.warning("anchor status keyword search failed: %s", type(exc).__name__)
        keyword_error = True

    vector_scores: dict[str, float] = {}
    vector_status = "ok"
    vector_batches = await asyncio.gather(*(
        embedding_engine.search_similar_with_status(
            angle, top_k=intent_policy["vector_top_k"]
        )
        for angle in query_angles
    ))
    for hits, status in vector_batches:
        if status != "ok":
            vector_status = status if status in {"empty", "timeout", "error", "circuit_open"} else "error"
            break
        for bucket_id, similarity in hits:
            if similarity > 0.5 and similarity > vector_scores.get(bucket_id, 0.0):
                vector_scores[bucket_id] = similarity

    entity_top_k, entity_weight = _entity_recall_settings()
    entity_store = _get_entity_store(initialize=False)
    entity_by_id: dict[str, dict] = {}
    entity_ranked: list[tuple[str, float]] = []
    for bucket_id, score in raw_entity_ranked[:entity_top_k]:
        try:
            bucket = keyword_by_id.get(bucket_id) or await bucket_mgr.get(bucket_id)
            if not bucket or not _passes_nonkeyword_recall_filters(
                bucket,
                world_filter_set=wf_set,
            ):
                continue
            if entity_store is None or not entity_store.link_is_current(
                bucket_id, bucket.get("content", "")
            ):
                continue
            if not _filter_z_fact_candidates(
                [bucket],
                query=recall_query,
                intent=intent_policy["intent"],
            ):
                continue
            entity_by_id[bucket_id] = bucket
            entity_ranked.append((bucket_id, score))
        except Exception as exc:
            logger.warning(
                "anchor status entity candidate skipped: %s",
                type(exc).__name__,
            )

    matches: list[dict] = []
    if vector_status == "ok" and not keyword_error:
        keyword_ranked = [(bucket["id"], bucket.get("score", 0)) for bucket in keyword_by_id.values()]
        channels = [
            (keyword_ranked, intent_policy["keyword_weight"]),
            (list(vector_scores.items()), intent_policy["vector_weight"]),
        ]
        if entity_ranked and entity_weight > 0:
            channels.append((entity_ranked, entity_weight))
        fused_pairs = rrf_fuse_channels(
            channels,
            k=(config.get("rrf", {}) or {}).get("k", 60),
        )
        for bucket_id, fused_score in fused_pairs[:recall_limit]:
            bucket = keyword_by_id.get(bucket_id) or entity_by_id.get(bucket_id)
            bucket = bucket or await bucket_mgr.get(bucket_id)
            if not bucket or not _passes_nonkeyword_recall_filters(
                bucket,
                world_filter_set=wf_set,
            ):
                continue
            if not _filter_z_fact_candidates(
                [bucket],
                query=recall_query,
                intent=intent_policy["intent"],
            ):
                continue
            bucket["score"] = round(fused_score * 1000, 2)
            matches.append(bucket)
        matches = await _ds_filter_candidates(
            recall_query,
            matches,
            mode="search",
            max_results=2,
            force_keep_ids=_exact_retrieval_key_ids(recall_query, matches),
        )

    has_evidence = bool(matches) if vector_status == "ok" and not keyword_error else False
    if keyword_error and vector_status == "ok":
        vector_status = "error"
    record = {
        "ts": now_iso(),
        "query_len": len(query),
        "intent": intent_policy.get("intent", "default"),
        "angle_count": len(query_angles),
        "query_expansion_skipped_hot_path": bool(qe_cfg.get("enabled", False)),
        "vector_status": vector_status,
        "keyword_candidate_count": len(keyword_by_id),
        "vector_candidate_count": len(vector_scores),
        "entity_candidate_count": len(entity_ranked),
        "entity_query_canonicalized": recall_query != query,
        "final_candidate_count": len(matches),
        "has_evidence": has_evidence,
        "timing_ms": round((time.monotonic() - started) * 1000, 2),
    }
    _append_recall_status_trace(record)
    return record


@mcp.custom_route("/api/anchor-status", methods=["GET"])
async def api_anchor_status(request):
    """Content-free Anchor health/evidence probe for twin cold-store arbitration."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    if len(query) > 500:
        return JSONResponse({"error": "q too long"}, status_code=400)
    try:
        result = await _probe_anchor_status(query)
        return JSONResponse(result)
    except Exception as exc:
        logger.warning("anchor status probe failed: %s", type(exc).__name__)
        return JSONResponse({
            "vector_status": "error",
            "has_evidence": False,
            "final_candidate_count": 0,
        })


@mcp.custom_route("/api/review_queue", methods=["GET"])
async def api_review_queue(request):
    """Return the real pending review queue; never substitute demo rows."""
    from starlette.responses import JSONResponse
    kind = (request.query_params.get("kind") or "").strip().lower()
    if kind and kind not in (
        KIND_CLOTHING,
        KIND_RELATION,
        KIND_Z_CONFLICT,
        KIND_METABOLISM,
        KIND_E_PROPOSAL,
    ):
        return JSONResponse({"error": "invalid kind"}, status_code=400)
    try:
        items = await asyncio.to_thread(_get_review_queue().list_pending, kind or None)
    except Exception as exc:
        logger.warning("review queue read failed: %s", type(exc).__name__)
        return JSONResponse({"error": "review queue unavailable"}, status_code=503)
    return JSONResponse({"items": items, "total": len(items), "status": "ready"})


def _review_write_api_enabled() -> bool:
    return bool(os.environ.get("OMBRE_API_TOKEN", "").strip())


@mcp.custom_route("/api/review_queue/resolve", methods=["POST"])
async def api_review_queue_resolve(request):
    """Acknowledge/reject a pending row without applying memory mutations."""
    from starlette.responses import JSONResponse
    if not _review_write_api_enabled():
        return JSONResponse({"error": "OMBRE_API_TOKEN required for review queue writes"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    key = str(body.get("key") or "").strip()
    status = str(body.get("status") or "").strip()
    verdict_note = str(body.get("verdict_note") or "").strip()[:500]
    if not key:
        return JSONResponse({"error": "key required"}, status_code=400)
    if not rest_resolve_status_allowed(status):
        return JSONResponse({
            "error": "REST resolve only supports reviewed/rejected; applied requires an explicit memory transaction",
        }, status_code=409)
    try:
        changed = await asyncio.to_thread(
            _get_review_queue().resolve,
            key,
            status,
            verdict_note=verdict_note,
        )
    except Exception as exc:
        logger.warning("review queue resolve failed: %s", type(exc).__name__)
        return JSONResponse({"error": "review queue unavailable"}, status_code=503)
    if not changed:
        return JSONResponse({"error": "pending review item not found"}, status_code=404)
    return JSONResponse({
        "ok": True,
        "key": key,
        "status": status,
        "memory_mutated": False,
    })


@mcp.custom_route("/api/review_queue/candidate", methods=["POST"])
async def api_review_queue_candidate(request):
    """Preview a Z pair by default; explicit mode=apply only queues pending."""
    from starlette.responses import JSONResponse
    if not _review_write_api_enabled():
        return JSONResponse({"error": "OMBRE_API_TOKEN required for review queue writes"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    return await _submit_review_queue_candidate(body)


async def _submit_review_queue_candidate(body: dict):
    """Validate one pair and either preview it or enqueue a pending decision."""
    from starlette.responses import JSONResponse
    mode = str(body.get("mode") or "dry-run").strip().lower()
    if mode not in {"dry-run", "apply"}:
        return JSONResponse(
            {"error": "mode must be dry-run or apply"},
            status_code=400,
        )
    current_id = str(body.get("current_bucket_id") or "").strip()
    historical_id = str(body.get("historical_bucket_id") or "").strip()
    fact_key = registered_fact_key(body.get("fact_key"), _fact_slot_registry())
    if fact_key is None:
        return JSONResponse(
            {"error": "fact_key must be registered in config.fact_slots.registry"},
            status_code=400,
        )
    id_pattern = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
    if not id_pattern.fullmatch(current_id) or not id_pattern.fullmatch(historical_id):
        return JSONResponse({"error": "invalid bucket id"}, status_code=400)
    if current_id == historical_id:
        return JSONResponse({"error": "bucket ids must differ"}, status_code=400)
    current = await bucket_mgr.get(current_id)
    historical = await bucket_mgr.get(historical_id)
    if not current or not historical:
        return JSONResponse({"error": "bucket not found"}, status_code=404)
    validation_error = _z_pair_validation_error(current, historical, fact_key)
    if validation_error:
        return JSONResponse({"error": validation_error}, status_code=409)

    def _name(bucket):
        metadata = bucket.get("metadata") if isinstance(bucket.get("metadata"), dict) else {}
        return str(metadata.get("name") or "")[:160]

    entry = make_z_pair_entry(
        current_id,
        historical_id,
        fact_key=fact_key,
        current_name=_name(current),
        historical_name=_name(historical),
        reason=str(body.get("reason") or "cross_bucket_currentness")[:160],
        source=str(body.get("source") or "z_conflict_review")[:80],
    )
    if mode == "dry-run":
        existing = await _await_daemon_thread(
            lambda: _get_review_queue().get(entry["key"])
        )
        return JSONResponse({
            "ok": True,
            "mode": mode,
            "key": entry["key"],
            "status": "preview",
            "candidate": entry,
            "existing_status": existing.get("status") if existing else None,
            "queue_mutated": False,
            "memory_mutated": False,
        })

    try:
        added = await _await_daemon_thread(
            lambda: _get_review_queue().enqueue(entry)
        )
        durable = await _await_daemon_thread(
            lambda: _get_review_queue().get(entry["key"])
        )
    except Exception as exc:
        logger.warning("review queue candidate enqueue failed: %s", type(exc).__name__)
        return JSONResponse({"error": "review queue unavailable"}, status_code=503)
    if not durable or durable.get("status") != "pending":
        return JSONResponse({
            "error": (
                "the same Z candidate was already resolved; "
                "create a new pair instead of reviving it"
            ),
            "key": entry["key"],
            "existing_status": durable.get("status") if durable else None,
            "queue_mutated": False,
            "memory_mutated": False,
        }, status_code=409)
    return JSONResponse({
        "ok": True,
        "mode": mode,
        "key": entry["key"],
        "status": "pending",
        "added": added,
        "queue_mutated": added,
        "memory_mutated": False,
    })


@mcp.custom_route("/api/review_queue/apply-lifecycle", methods=["POST"])
async def api_review_queue_apply_lifecycle(request):
    """Apply one pending pair after an explicit named human approval."""
    from starlette.responses import JSONResponse
    if not _review_write_api_enabled():
        return JSONResponse({"error": "OMBRE_API_TOKEN required for review queue writes"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    return await _apply_review_queue_lifecycle(body)


async def _apply_review_queue_lifecycle(body):
    from starlette.responses import JSONResponse
    key = str(body.get("key") or "").strip()
    reviewer = str(body.get("reviewer") or "").strip()
    verdict_note = str(body.get("verdict_note") or "").strip()[:500]
    if not key:
        return JSONResponse({"error": "key required"}, status_code=400)
    if not reviewer:
        return JSONResponse(
            {"error": "reviewer required for explicit Z approval"},
            status_code=400,
        )
    try:
        result = await _await_daemon_thread(
            lambda: _get_z_lifecycle_transaction().apply(
                key,
                reviewer=reviewer,
                verdict_note=verdict_note,
                validate_pair=_z_pair_validation_error,
            )
        )
    except ZLifecycleNotFound as exc:
        return JSONResponse({
            "error": str(exc),
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=404)
    except (ValueError, ZLifecycleStateError) as exc:
        return JSONResponse({
            "error": str(exc),
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=409)
    except Exception as exc:
        logger.exception(
            "Z lifecycle transaction failed: %s",
            type(exc).__name__,
        )
        # 2026-08-19 review P2: the receipt must follow the durable ledger.  If the
        # review queue already recorded `applied`, the approval committed even
        # though a trailing step blew up (recovery finishes the targets on the
        # next start); answering "nothing changed" here would be a lie.
        durable = None
        try:
            durable = _get_z_lifecycle_transaction().review_queue.get(key)
        except Exception:  # noqa: BLE001 — best effort, never mask the original error
            durable = None
        if durable and durable.get("status") == "applied":
            return JSONResponse({
                "ok": True,
                "key": key,
                "status": "applied",
                "changed": True,
                "memory_mutated": True,
                "queue_mutated": True,
                "recovery_pending": True,
                "error": f"trailing step failed after durable commit: {type(exc).__name__}",
            })
        return JSONResponse({
            "error": "Z lifecycle transaction failed closed",
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=503)
    return JSONResponse({
        "ok": True,
        **result,
        "memory_mutated": result["changed"],
        "queue_mutated": result["changed"],
    })


@mcp.custom_route("/api/review_queue/apply-relation", methods=["POST"])
async def api_review_queue_apply_relation(request):
    """Apply one dangerous pending edge after explicit named approval."""
    from starlette.responses import JSONResponse
    if not _review_write_api_enabled():
        return JSONResponse(
            {"error": "OMBRE_API_TOKEN required for review queue writes"},
            status_code=503,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    return await _apply_review_queue_relation(body)


async def _apply_review_queue_relation(body):
    from starlette.responses import JSONResponse
    key = str(body.get("key") or "").strip()
    reviewer = str(body.get("reviewer") or "").strip()
    verdict_note = str(body.get("verdict_note") or "").strip()[:500]
    if not key:
        return JSONResponse({"error": "key required"}, status_code=400)
    if not reviewer:
        return JSONResponse(
            {"error": "reviewer required for explicit relation approval"},
            status_code=400,
        )
    try:
        result = await _await_daemon_thread(
            lambda: _get_relation_approval_transaction().apply(
                key,
                reviewer=reviewer,
                verdict_note=verdict_note,
            )
        )
    except RelationApprovalNotFound as exc:
        return JSONResponse({
            "error": str(exc),
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=404)
    except (ValueError, RelationApprovalStateError) as exc:
        return JSONResponse({
            "error": str(exc),
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=409)
    except Exception as exc:
        logger.exception(
            "relation approval transaction failed: %s",
            type(exc).__name__,
        )
        return JSONResponse({
            "error": "relation approval transaction failed closed",
            "memory_mutated": False,
            "queue_mutated": False,
        }, status_code=503)
    return JSONResponse({
        "ok": True,
        **result,
        "memory_mutated": result["changed"],
        "queue_mutated": result["queue_changed"],
    })


@mcp.custom_route("/api/review_queue/apply-protected-overlay", methods=["POST"])
async def api_review_queue_apply_protected_overlay(request):
    """Retired: protected/narrative memories are outside Z currentness."""
    from starlette.responses import JSONResponse
    if not _review_write_api_enabled():
        return JSONResponse({"error": "OMBRE_API_TOKEN required for review queue writes"}, status_code=503)
    return JSONResponse({
        "error": "protected memories are exempt from Z currentness filtering",
        "memory_mutated": False,
        "retrieval_policy_mutated": False,
    }, status_code=409)


@mcp.custom_route("/api/e-axis/shadow", methods=["POST"])
async def api_e_axis_shadow(request):
    """Persist a strictly validated E annotation outside the recall corpus."""
    from starlette.responses import JSONResponse

    if not _review_write_api_enabled():
        return JSONResponse({"error": "OMBRE_API_TOKEN required for E shadow writes"}, status_code=503)
    e_cfg = config.get("e_axis_shadow", {}) or {}
    if e_cfg.get("enabled") is not True:
        return JSONResponse({"error": "E shadow is disabled"}, status_code=409)
    try:
        body = strict_json_loads(await request.body())
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    required = {
        "bucket_id",
        "source_digest",
        "provider",
        "scorer",
        "model",
        "rubric_version",
        "run_id",
        "trigger_reason",
        "score",
    }
    if type(body) is not dict or set(body) != required:
        return JSONResponse({"error": "exact E shadow request schema required"}, status_code=400)

    bucket_id = str(body.get("bucket_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", bucket_id):
        return JSONResponse({"error": "invalid bucket_id"}, status_code=400)
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "bucket not found"}, status_code=404)
    current_digest = hashlib.sha256(
        str(bucket.get("content") or "").encode("utf-8")
    ).hexdigest()

    scorer = body.get("scorer")
    provider = body.get("provider")
    model = body.get("model")
    rubric_version = body.get("rubric_version")
    run_id = body.get("run_id")
    trigger_reason = body.get("trigger_reason")
    manual_source_run_id = "manual:" + hashlib.sha256(
        str(run_id or "").encode("utf-8")
    ).hexdigest()
    supplied_digest = str(body.get("source_digest") or "").strip().lower()
    store = _get_e_axis_shadow_store()
    if supplied_digest != current_digest:
        failure = build_failure_record(
            bucket_id=bucket_id,
            source_digest=current_digest,
            source_kind="manual_bucket",
            source_run_id=manual_source_run_id,
            provider=provider,
            scorer=scorer,
            model=model,
            rubric_version=rubric_version,
            run_id=run_id,
            trigger_reason=trigger_reason,
            category="source_digest.mismatch",
            retryable=False,
        )
        try:
            await asyncio.to_thread(store.append, failure)
        except Exception as exc:
            logger.warning("E shadow failure ledger unavailable: %s", type(exc).__name__)
        return JSONResponse({"error": "source_digest does not match current content"}, status_code=409)

    min_confidence = normalize_min_confidence(e_cfg.get("min_confidence", 0.3))
    if min_confidence is None:
        return JSONResponse(
            {"error": "invalid E shadow min_confidence config"},
            status_code=503,
        )
    annotation, error = build_shadow_annotation(
        bucket_id=bucket_id,
        source_digest=current_digest,
        source_kind="manual_bucket",
        source_run_id=manual_source_run_id,
        provider=provider,
        scorer=scorer,
        model=model,
        rubric_version=rubric_version,
        run_id=run_id,
        trigger_reason=trigger_reason,
        score=body.get("score"),
        min_confidence=min_confidence,
    )
    if error:
        failure = build_failure_record(
            bucket_id=bucket_id,
            source_digest=current_digest,
            source_kind="manual_bucket",
            source_run_id=manual_source_run_id,
            provider=provider,
            scorer=scorer,
            model=model,
            rubric_version=rubric_version,
            run_id=run_id,
            trigger_reason=trigger_reason,
            category=error,
            retryable=False,
        )
        try:
            await asyncio.to_thread(store.append, failure)
        except Exception as exc:
            logger.warning("E shadow failure ledger unavailable: %s", type(exc).__name__)
        return JSONResponse({"error": error, "shadow_only": True}, status_code=422)

    try:
        added = await asyncio.to_thread(store.append, annotation)
    except Exception as exc:
        logger.warning("E shadow ledger unavailable: %s", type(exc).__name__)
        return JSONResponse({"error": "E shadow ledger unavailable"}, status_code=503)
    return JSONResponse({
        "ok": True,
        "annotation_key": annotation["annotation_key"],
        "added": added,
        "shadow_only": True,
        "affects_ranking": False,
        "memory_mutated": False,
    })


@mcp.custom_route("/", methods=["GET"])
async def root_redirect(request):
    """根路径跳转到 /dashboard，避免存根书签打不开(404)。"""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review."""
    from starlette.responses import JSONResponse
    try:
        limit = int(request.query_params.get("limit", "50"))
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(
            key=lambda b: event_at_from_metadata(b["metadata"]) or "",
            reverse=True,
        )
        results = []
        for b in all_buckets[:limit]:
            results.append({
                "id": b["id"],
                "name": b["metadata"].get("name", ""),
                "content": redact_text(b["content"])[:300],
                "type": b["metadata"].get("type", ""),
                "domain": b["metadata"].get("domain", []),
                "tags": b["metadata"].get("tags", []),
                "importance": b["metadata"].get("importance", 5),
                "event_at": event_at_from_metadata(b["metadata"]) or "",
                "recorded_at": b["metadata"].get("recorded_at", ""),
                "date_precision": b["metadata"].get("date_precision", "unknown"),
                "date_source": b["metadata"].get("date_source", ""),
                "date_confidence": b["metadata"].get("date_confidence"),
                "created": event_at_from_metadata(b["metadata"]) or "",
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                if not await bucket_mgr.delete(bid):
                    raise RuntimeError(f"delete failed: {bid}")
                _unlink_bucket_entities(bid)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# =============================================================
# Twin REST endpoints — bridge for Telegram bot (and other thin frontends)
# Twin REST 接口 —— 给 Telegram bot（及其他薄前端）用的桥接
# =============================================================
def _breath_deadline_sec() -> float:
    raw = os.getenv(
        "OMBRE_BREATH_DEADLINE_SEC",
        str(config.get("breath_deadline_sec", 11.0)),
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 11.0
    return max(0.1, min(value, 13.0))


@mcp.custom_route("/api/breath", methods=["POST"])
async def api_breath(request):
    """HTTP bridge to the read-only ``breath`` tool.

    The bridge deliberately exposes only the search arguments needed by thin
    frontends.  Images and the external body-state block are always disabled,
    so the JSON response is text-only and cannot carry image payloads or mutate
    body state.  Search-mode retrieval is bucket-read-only, just like a direct
    MCP ``breath`` call: neither path touches bucket activity metadata.
    """
    from starlette.responses import JSONResponse

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)

    query = str(body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)

    def _int_arg(name, default):
        try:
            return int(body.get(name, default))
        except (TypeError, ValueError):
            return default

    def _float_arg(name, default=-1.0):
        try:
            return float(body.get(name, default))
        except (TypeError, ValueError):
            return default

    requested_policy = str(body.get("policy") or "search").strip().lower()
    recall_policy = _normalize_anchor_recall_policy(requested_policy)
    timing_token = begin_recall_timing()
    breath_task = None
    partial = False
    try:
        try:
            breath_task = asyncio.create_task(breath(
                query=query,
                max_tokens=_int_arg("max_tokens", BREATH_DEFAULT_MAX_TOKENS),
                domain=str(body.get("domain") or ""),
                valence=_float_arg("valence"),
                arousal=_float_arg("arousal"),
                max_results=_int_arg("max_results", BREATH_DEFAULT_MAX_RESULTS),
                world=str(body.get("world") or ""),
                relation_depth=_int_arg("relation_depth", 1),
                since=str(body.get("since") or ""),
                until=str(body.get("until") or ""),
                session_id=str(body.get("session_id") or ""),
                policy=recall_policy,
                include_images=False,
                include_body_state=False,
                reset_body_state=False,
            ))
            done, _pending = await asyncio.wait(
                {breath_task},
                timeout=_breath_deadline_sec(),
            )
            if done:
                result = breath_task.result()
            else:
                partial = True
                result = get_recall_partial_result().strip() or "未找到相关记忆。"
                breath_task.cancel()
                try:
                    await breath_task
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            if breath_task is not None and not breath_task.done():
                breath_task.cancel()
                try:
                    await breath_task
                except asyncio.CancelledError:
                    pass
            timing = finish_recall_timing(status="cancelled", partial=True)
            logger.info("breath_timing=%s", json.dumps(timing, sort_keys=True))
            raise
        except Exception:
            timing = finish_recall_timing(status="error", partial=False)
            logger.exception("HTTP breath bridge failed")
            logger.info("breath_timing=%s", json.dumps(timing, sort_keys=True))
            return JSONResponse(
                {"error": "breath failed", "partial": False, "timing": timing},
                status_code=500,
            )

        timing = finish_recall_timing(
            status="deadline" if partial else "ok",
            partial=partial,
        )
        logger.info("breath_timing=%s", json.dumps(timing, sort_keys=True))
    finally:
        reset_recall_timing(timing_token)

    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        text = "\n".join(
            value for item in result
            if isinstance((value := getattr(item, "text", None)), str)
        )
    else:
        text = str(result)
    return JSONResponse({
        "raw": text,
        "policy": recall_policy,
        "partial": partial,
        "timing": timing,
    })


@mcp.custom_route("/api/recall-receipt", methods=["POST"])
async def api_recall_receipt(request):
    """Idempotently activate only memories committed to a final model prompt.

    ``breath`` remains read-only.  This separate write endpoint is best-effort
    from the caller's perspective: a failed receipt must never suppress the
    recall text or the assistant response.  No query or memory content enters
    the receipt ledger.
    """
    from starlette.responses import JSONResponse

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    event_id = str(body.get("event_id") or "").strip()
    raw_ids = body.get("bucket_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [part.strip() for part in raw_ids.split(",") if part.strip()]
    try:
        bucket_ids = normalize_bucket_ids(raw_ids)
        store = _get_recall_receipt_store()
        begun = store.begin(
            event_id,
            bucket_ids,
            str(body.get("source") or "twin_prompt_injection"),
        )
    except RecallReceiptConflict:
        return JSONResponse({"error": "event_id payload conflict"}, status_code=409)
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("Recall receipt ledger unavailable")
        return JSONResponse({"error": "receipt ledger unavailable"}, status_code=503)

    applied_now = 0
    failed = []
    actor_key = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:24]
    for bucket_id in begun["pending"]:
        try:
            if await bucket_mgr.get(bucket_id) is None:
                raise KeyError("bucket not found")
            # Anchor spreads heat only over explicit conductive edges.  Ombre's
            # legacy touch ripple is temporal rather than conductive, so this
            # first adapter deliberately activates the injected root only.
            await bucket_mgr.touch(
                bucket_id,
                actor=f"system:recall_receipt:{actor_key}",
                ripple=False,
                raise_on_error=True,
            )
            store.mark_applied(event_id, bucket_id)
            applied_now += 1
        except Exception as exc:
            store.mark_failed(event_id, bucket_id, exc)
            failed.append({"bucket_id": bucket_id, "error": type(exc).__name__})
    status = store.status(event_id)
    return JSONResponse({
        "ok": not failed,
        "event_id": event_id,
        "duplicate": bool(begun["duplicate"]),
        "applied_now": applied_now,
        "failed": failed,
        **status,
    })


@mcp.custom_route("/api/hold", methods=["POST"])
async def api_hold(request):
    """HTTP bridge to hold tool. Body: {content, tags?, importance?, pinned?, source?,
    domain?, feel?, chord_tag?, valence?, arousal?, source_bucket?}.
    HTTP 桥接 hold 工具。source 会作为额外标签合入 tags。
    feel/chord_tag/valence/arousal/source_bucket 透传给 hold——让 server 侧能替哥哥落第一人称
    feel 桶（如逛 X 的体验沉进海马体，2026-06-04 接 C）。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    content = str(body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)

    raw_tags = body.get("tags") or []
    if isinstance(raw_tags, list):
        tag_parts = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tag_parts = [t.strip() for t in str(raw_tags).split(",") if t.strip()]
    source = str(body.get("source") or "").strip()
    if source and source not in tag_parts:
        tag_parts.append(source)
    tags_csv = ",".join(tag_parts)

    try:
        importance = int(body.get("importance") or 5)
    except (TypeError, ValueError):
        importance = 5
    pinned = bool(body.get("pinned"))

    raw_domain = body.get("domain") or ""
    if isinstance(raw_domain, list):
        domain_csv = ",".join(str(d).strip() for d in raw_domain if str(d).strip())
    else:
        domain_csv = ",".join(d.strip() for d in str(raw_domain).split(",") if d.strip())

    feel = bool(body.get("feel"))
    chord_tag = str(body.get("chord_tag") or "").strip()
    source_bucket = str(body.get("source_bucket") or "").strip()

    def _num(key, default=-1.0):
        try:
            return float(body.get(key))
        except (TypeError, ValueError):
            return default

    valence = _num("valence")
    arousal = _num("arousal")

    try:
        result = await hold(
            content=content,
            tags=tags_csv,
            importance=importance,
            pinned=pinned,
            domain=domain_csv,
            feel=feel,
            chord_tag=chord_tag,
            valence=valence,
            arousal=arousal,
            source_bucket=source_bucket,
        )
        return JSONResponse({"result": result})
    except Exception as e:
        logger.error(f"/api/hold failed / 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/briefing", methods=["GET"])
async def api_briefing(request):
    """HTTP bridge to briefing tool. Query: ?max_chars=&domain=&pinned_only=&format=
    HTTP 桥接 briefing 工具。format=text(默认)返回纯文本简报;
    format=json 返回 {slots[](tier=0 原文), briefing(LLM 压缩文)} 结构(#4 核心画像分离)。"""
    from starlette.responses import PlainTextResponse, JSONResponse, Response
    try:
        try:
            max_chars = int(request.query_params.get("max_chars", 1500))
        except ValueError:
            max_chars = 1500
        domain = request.query_params.get("domain", "")
        pinned_only = request.query_params.get("pinned_only", "").lower() in ("1", "true", "yes")
        session_id = request.query_params.get("session_id", "")
        include_body_state = request.query_params.get("include_body_state", "true").lower() not in ("0", "false", "no", "off")
        reset_body_state = request.query_params.get("reset_body_state", "").lower() in ("1", "true", "yes", "on")
        deterministic = request.query_params.get("deterministic", "").lower() in ("1", "true", "yes", "on")
        fmt = (request.query_params.get("format", "text") or "text").lower()
        if fmt not in ("text", "json"):
            fmt = "text"
        text = await briefing(
            max_chars=max_chars,
            domain=domain,
            pinned_only=pinned_only,
            session_id=session_id,
            include_body_state=include_body_state,
            reset_body_state=reset_body_state,
            format=fmt,
            deterministic=deterministic,
        )
        if fmt == "json":
            # briefing() 已 json.dumps 出 UTF-8 字符串；整体过 redact 兜底（[REDACTED] 不含
            # json 特殊字符、不破坏结构），覆盖 anchor label 等逐处未接的角落。
            return Response(content=redact_text(text), media_type="application/json")
        return PlainTextResponse(redact_text(text))
    except Exception as e:
        logger.error(f"/api/briefing failed / 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Twin queue: bot ↔ CC asynchronous bridge (PoC: jsonl file queue)
# Twin 队列：bot ↔ CC 异步桥接（PoC：jsonl 文件队列）
#
# 朝灯发消息进 Telegram → bot → POST /api/inbox → inbox.jsonl
# CC 调 twin_pull MCP → 读未读 → 我看 → 我回 → 调 twin_send → outbox.jsonl
# bot 后台轮询 GET /api/outbox?after=<id> → 推送回 Telegram
#
# 文件位置：{buckets_dir}/twin/inbox.jsonl, outbox.jsonl
# 每行一条 JSON：{id, ts, source, text, user_id?, read?}
# =============================================================
_TWIN_DIR = os.path.join(config["buckets_dir"], "twin")
os.makedirs(_TWIN_DIR, exist_ok=True)
_TWIN_INBOX = os.path.join(_TWIN_DIR, "inbox.jsonl")
_TWIN_OUTBOX = os.path.join(_TWIN_DIR, "outbox.jsonl")
_twin_lock = asyncio.Lock()


def _twin_append_sync(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _twin_read_all_sync(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _twin_rewrite_sync(path: str, records: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# =============================================================
# Tool: twin_pull — 我（CC）拉 inbox 看朝灯发了什么
# =============================================================
@mcp.tool()
async def twin_pull(unread_only: bool = True, mark_read: bool = True, limit: int = 20) -> str:
    """从 Telegram 端拉取消息。unread_only=True 只看未读,mark_read=True 拉取后标记已读。返回最近的消息。
    朝灯通过 Telegram bot 发的话进 inbox,CC 开窗调这个工具看她说了什么。"""
    async with _twin_lock:
        records = _twin_read_all_sync(_TWIN_INBOX)
        if unread_only:
            picked = [r for r in records if not r.get("read")]
        else:
            picked = list(records)
        picked = picked[-max(1, limit):]
        if mark_read and unread_only and picked:
            picked_ids = {r["id"] for r in picked}
            for r in records:
                if r.get("id") in picked_ids:
                    r["read"] = True
            _twin_rewrite_sync(_TWIN_INBOX, records)

    if not picked:
        return "(inbox 空 / 无未读)"
    lines = []
    for r in picked:
        ts = r.get("ts", "")
        src = r.get("source", "?")
        txt = r.get("text", "")
        lines.append(f"[{ts}] {src}: {txt}")
    return "\n".join(lines)


# =============================================================
# Tool: twin_send — 我（CC）回话写到 outbox,bot 轮询拉走推到 Telegram
# =============================================================
@mcp.tool()
async def twin_send(text: str, to: str = "telegram") -> str:
    """回复朝灯,消息写到 outbox。bot 后台轮询会推到 Telegram。to 默认 telegram。"""
    text = (text or "").strip()
    if not text:
        return "空消息,未发送。"
    rec = {
        "id": uuid4().hex[:12],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "to": to,
        "text": text,
    }
    async with _twin_lock:
        _twin_append_sync(_TWIN_OUTBOX, rec)
    return f"✉️ → {to} ({rec['id']})"


# =============================================================
# REST: bot 写入 inbox（朝灯发的消息）
# =============================================================
@mcp.custom_route("/api/inbox", methods=["POST"])
async def api_inbox_post(request):
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    rec = {
        "id": uuid4().hex[:12],
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": str(body.get("source") or "telegram"),
        "user_id": str(body.get("user_id") or ""),
        "text": text,
        "read": False,
    }
    async with _twin_lock:
        _twin_append_sync(_TWIN_INBOX, rec)
    return JSONResponse({"id": rec["id"], "ts": rec["ts"]})


# =============================================================
# REST: bot 轮询 outbox（CC 写的回复）
# =============================================================
@mcp.custom_route("/api/outbox", methods=["GET"])
async def api_outbox_get(request):
    """Query: ?after=<id> → 返回 id 之后的所有 outbox 消息。
    after 为空时返回全部（首次连接用）。"""
    from starlette.responses import JSONResponse
    after = request.query_params.get("after", "")
    async with _twin_lock:
        records = _twin_read_all_sync(_TWIN_OUTBOX)
    if after:
        cut = -1
        for i, r in enumerate(records):
            if r.get("id") == after:
                cut = i
                break
        if cut >= 0:
            records = records[cut + 1:]
    return JSONResponse({"messages": records})


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    try:
        recovered_z = _get_z_lifecycle_transaction().recover()
        if recovered_z:
            logger.warning(
                "Recovered %d interrupted Z lifecycle transaction(s)",
                len(recovered_z),
            )
    except Exception:
        logger.exception("Z lifecycle recovery failed; refusing to start")
        raise
    try:
        recovered_relations = _get_relation_approval_transaction().recover()
        if recovered_relations:
            logger.warning(
                "Recovered %d interrupted relation approval transaction(s)",
                len(recovered_relations),
            )
    except Exception:
        logger.exception("Relation approval recovery failed; refusing to start")
        raise

    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    # Seed initialization is an explicit startup mutation, never a breath
    # side effect.  This makes audited aliases available immediately after a
    # restart while keeping every recall call filesystem-read-only.
    entity_cfg = config.get("entities", {}) or {}
    if entity_cfg.get("enabled", True) and (entity_cfg.get("seeds") or []):
        if _get_entity_store(initialize=True) is None:
            logger.warning("Entity seed initialization failed; legacy recall remains active")
        else:
            logger.info("Entity alias sidecar initialized from audited seeds")

    if transport in ("sse", "streamable-http"):
        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        # 两种跑法，用 OMBRE_LOOPBACK_ONLY 明确分开：
        #   · 容器（NAS，默认）：跟原来一样 0.0.0.0:8000，宿主 compose 再把 8000 映射到
        #     127.0.0.1:8000。**不能**拿 OMBRE_BIND_ADDRESS 当开关——compose 会把 .env 里给
        #     宿主端口用的 OMBRE_BIND_ADDRESS=127.0.0.1 原样传进容器，8/18 第一次部署就是
        #     这么死的：容器里只听 127.0.0.1:18080，宿主映射的 8000 没人接。
        #   · VPS 灾备单机（run-local.sh 设 OMBRE_LOOPBACK_ONLY=1）：只许 loopback，端口默认 18080。
        loopback_only = os.environ.get("OMBRE_LOOPBACK_ONLY", "").strip().lower() in {"1", "true", "yes"}
        if loopback_only:
            bind_host = os.environ.get("OMBRE_BIND_ADDRESS", "127.0.0.1").strip()
            if bind_host not in {"127.0.0.1", "::1", "localhost"}:
                raise RuntimeError(
                    "VPS disaster-recovery runtime only permits a loopback bind"
                )
            try:
                host_port = int(os.environ.get("OMBRE_HOST_PORT", "18080"))
            except ValueError as exc:
                raise RuntimeError("OMBRE_HOST_PORT must be an integer") from exc
            if not 1 <= host_port <= 65535:
                raise RuntimeError("OMBRE_HOST_PORT is outside the valid range")
        else:
            bind_host, host_port = "0.0.0.0", 8000
        keepalive_host = "localhost" if bind_host == "0.0.0.0" else bind_host

        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get(
                            f"http://{keepalive_host}:{host_port}/health",
                            timeout=5,
                        )
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()

        # FastMCP(lifespan=_server_lifespan) does NOT run on this path:
        # streamable_http_app()'s ASGI lifespan only manages the session
        # manager, so the briefing prebuild worker never started (verified
        # 2026-08-11: container log has "session manager started" but no
        # "Briefing background pre-generation started"). Wrap the app's own
        # lifespan so the worker starts under uvicorn too.
        _transport_lifespan = _app.router.lifespan_context

        @asynccontextmanager
        async def _uvicorn_lifespan(app):
            async with _transport_lifespan(app):
                await pg_mirror_worker.start()
                await _start_briefing_cache_refresh()
                try:
                    yield
                finally:
                    await _stop_briefing_cache_refresh()
                    await pg_mirror_worker.stop()

        _app.router.lifespan_context = _uvicorn_lifespan

        # --- Network authentication boundary ---
        # /api/* uses OMBRE_API_TOKEN; MCP transports use OMBRE_MCP_TOKEN.
        # Both are mandatory for network mode.  /health remains anonymous so
        # container and tunnel health checks never need a secret.
        #
        # MCP uses a pure ASGI middleware: BaseHTTPMiddleware may buffer or
        # cancel streaming/SSE bodies and is therefore unsuitable here.
        _OMBRE_API_TOKEN = require_api_token()
        _OMBRE_MCP_TOKEN = require_mcp_token()

        _app.add_middleware(APIBearerAuthMiddleware, token=_OMBRE_API_TOKEN)
        _app.add_middleware(MCPBearerAuthMiddleware, token=_OMBRE_MCP_TOKEN)

        # CORS：前端不直连 Ombre（经 claude-twin 代理），故收紧到本机回环 —— 杀掉
        # 「任意网站用浏览器跨域读你记忆」的攻击面。MCP 是服务端到服务端、不吃浏览器 CORS。
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        logger.info(
            "CORS + /api + MCP Bearer 鉴权已启用 / network auth middleware enabled"
        )
        uvicorn.run(_app, host=bind_host, port=host_port)
    else:
        mcp.run(transport=transport)
