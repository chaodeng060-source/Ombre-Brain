# ============================================================
# Module: Memory Import Engine (import_memory.py)
# 模块：历史记忆导入引擎
#
# Imports conversation history from various platforms into OB.
# 将各平台对话历史导入 OB 记忆系统。
#
# Supports: Claude JSON, ChatGPT export, DeepSeek, Markdown, plain text
# 支持格式：Claude JSON、ChatGPT 导出、DeepSeek、Markdown、纯文本
#
# Features:
#   - Chunked processing with resume support
#   - Progress persistence (import_state.json)
#   - Raw preservation mode for special contexts
#   - Post-import frequency pattern detection
# ============================================================

import os
import json
import hashlib
import inspect
import logging
from datetime import datetime
from pathlib import Path

from utils import count_tokens_approx, normalize_event_at, now_iso
from redact import redact_embedding_input, redact_text  # 只抹 secret，不审查情感内容
from maintenance_barrier import MaintenanceBarrier
from storage_safety import advisory_file_lock, atomic_write_text

logger = logging.getLogger("ombre_brain.import")


IMPORT_STATE_SCHEMA_VERSION = 2
IMPORT_OUTPUT_MARKER_PREFIX = "ombre-import-v1"
CHUNK_STATUSES = {"pending", "running", "complete", "error", "deferred"}
OUTPUT_STATUSES = {"pending", "running", "complete", "error", "deferred"}


class ImportExtractionError(RuntimeError):
    """The provider response cannot be treated as a successful extraction."""


class ImportStorageError(RuntimeError):
    """A durable bucket or required index write did not complete."""


def _stable_json_digest(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chunk_identity(source_hash: str, chunk_index: int, chunk: dict) -> str:
    return _stable_json_digest({
        "contract": "ombre-import-chunk/v1",
        "source_hash": source_hash,
        "chunk_index": chunk_index,
        "extraction_input_digest": hashlib.sha256(
            _import_extraction_input(chunk["content"]).encode("utf-8")
        ).hexdigest(),
    })


def _output_identity(
    chunk_id: str,
    output_index: int,
    item: dict,
    preserve_raw: bool,
) -> str:
    return _stable_json_digest({
        "contract": "ombre-import-output/v1",
        "chunk_id": chunk_id,
        "output_index": output_index,
        "item": item,
        "preserve_raw": bool(preserve_raw),
    })


def _output_marker(output_id: str, action: str) -> str:
    if action not in {"created", "merged", "raw"}:
        raise ValueError(f"unsupported import output action: {action}")
    return f"{IMPORT_OUTPUT_MARKER_PREFIX}:{output_id}:{action}"


def _output_marker_prefix(output_id: str) -> str:
    return f"{IMPORT_OUTPUT_MARKER_PREFIX}:{output_id}:"


def _dedupe_tags(tags: list[str]) -> list[str]:
    result = []
    seen = set()
    for tag in tags:
        value = str(tag)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


# ============================================================
# Format Parsers — normalize any format to conversation turns
# 格式解析器 — 将任意格式标准化为对话轮次
# ============================================================

def _parse_claude_json(data: dict | list) -> list[dict]:
    """Parse Claude.ai export JSON → [{role, content, timestamp}, ...]"""
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        messages = conv.get("chat_messages", conv.get("messages", []))
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("text", msg.get("content", ""))
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if not content or not content.strip():
                continue
            role = msg.get("sender", msg.get("role", "user"))
            ts = msg.get("created_at", msg.get("timestamp", ""))
            turns.append({"role": role, "content": content.strip(), "timestamp": ts})
    return turns


def _parse_chatgpt_json(data: list | dict) -> list[dict]:
    """Parse ChatGPT export JSON → [{role, content, timestamp}, ...]"""
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        mapping = conv.get("mapping", {})
        if mapping:
            # ChatGPT uses a tree structure with mapping
            sorted_nodes = sorted(
                mapping.values(),
                key=lambda n: n.get("message", {}).get("create_time", 0) or 0,
            )
            for node in sorted_nodes:
                msg = node.get("message")
                if not msg or not isinstance(msg, dict):
                    continue
                content_parts = msg.get("content", {}).get("parts", [])
                content = " ".join(str(p) for p in content_parts if p)
                if not content.strip():
                    continue
                role = msg.get("author", {}).get("role", "user")
                ts = msg.get("create_time", "")
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts).isoformat()
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
        else:
            # Simpler format: list of messages
            messages = conv.get("messages", [])
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", msg.get("text", ""))
                if isinstance(content, dict):
                    content = " ".join(str(p) for p in content.get("parts", []))
                if not content or not content.strip():
                    continue
                role = msg.get("role", msg.get("author", {}).get("role", "user"))
                ts = msg.get("timestamp", msg.get("create_time", ""))
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
    return turns


def _parse_markdown(text: str) -> list[dict]:
    """Parse Markdown/plain text → [{role, content, timestamp}, ...]"""
    # Try to detect conversation patterns
    lines = text.split("\n")
    turns = []
    current_role = "user"
    current_content = []

    for line in lines:
        stripped = line.strip()
        # Detect role switches
        if stripped.lower().startswith(("human:", "user:", "你:", "我:")):
            if current_content:
                turns.append({"role": current_role, "content": "\n".join(current_content).strip(), "timestamp": ""})
            current_role = "user"
            content_after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            current_content = [content_after] if content_after else []
        elif stripped.lower().startswith(("assistant:", "claude:", "ai:", "gpt:", "bot:", "deepseek:")):
            if current_content:
                turns.append({"role": current_role, "content": "\n".join(current_content).strip(), "timestamp": ""})
            current_role = "assistant"
            content_after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            current_content = [content_after] if content_after else []
        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            turns.append({"role": current_role, "content": content, "timestamp": ""})

    # If no role patterns detected, treat entire text as one big chunk
    if not turns:
        turns = [{"role": "user", "content": text.strip(), "timestamp": ""}]

    return turns


def detect_and_parse(raw_content: str, filename: str = "") -> list[dict]:
    """
    Auto-detect format and parse to normalized turns.
    自动检测格式并解析为标准化的对话轮次。
    """
    ext = Path(filename).suffix.lower() if filename else ""

    # Try JSON first
    if ext in (".json", "") or raw_content.strip().startswith(("{", "[")):
        try:
            data = json.loads(raw_content)
            # Detect Claude vs ChatGPT format
            if isinstance(data, list):
                sample = data[0] if data else {}
            else:
                sample = data

            if isinstance(sample, dict):
                if "chat_messages" in sample:
                    return _parse_claude_json(data)
                if "mapping" in sample:
                    return _parse_chatgpt_json(data)
                if "messages" in sample:
                    # Could be either — try ChatGPT first, fall back to Claude
                    msgs = sample["messages"]
                    if msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
                        if isinstance(msgs[0]["content"], dict):
                            return _parse_chatgpt_json(data)
                    return _parse_claude_json(data)
                # Single conversation object with role/content messages
                if "role" in sample and "content" in sample:
                    return _parse_claude_json(data)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Fall back to markdown/text
    return _parse_markdown(raw_content)


# ============================================================
# Chunking — split turns into ~10k token windows
# 分窗 — 按对话轮次边界切为 ~10k token 窗口
# ============================================================

def chunk_turns(turns: list[dict], target_tokens: int = 10000) -> list[dict]:
    """
    Group conversation turns into chunks of ~target_tokens.
    Returns list of {content, timestamp_start, timestamp_end, turn_count}.
    按对话轮次边界将对话分为 ~target_tokens 大小的窗口。
    """
    chunks = []
    current_lines = []
    current_tokens = 0
    first_ts = ""
    last_ts = ""
    turn_count = 0

    for turn in turns:
        role_label = "用户" if turn["role"] in ("user", "human") else "AI"
        line = f"[{role_label}] {turn['content']}"
        line_tokens = count_tokens_approx(line)

        # If single turn exceeds target, split it
        if line_tokens > target_tokens * 1.5:
            # Flush current
            if current_lines:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "timestamp_start": first_ts,
                    "timestamp_end": last_ts,
                    "turn_count": turn_count,
                })
                current_lines = []
                current_tokens = 0
                turn_count = 0
                first_ts = ""

            # Add oversized turn as its own chunk
            chunks.append({
                "content": line,
                "timestamp_start": turn.get("timestamp", ""),
                "timestamp_end": turn.get("timestamp", ""),
                "turn_count": 1,
            })
            continue

        if current_tokens + line_tokens > target_tokens and current_lines:
            chunks.append({
                "content": "\n".join(current_lines),
                "timestamp_start": first_ts,
                "timestamp_end": last_ts,
                "turn_count": turn_count,
            })
            current_lines = []
            current_tokens = 0
            turn_count = 0
            first_ts = ""

        if not first_ts:
            first_ts = turn.get("timestamp", "")
        last_ts = turn.get("timestamp", "")
        current_lines.append(line)
        current_tokens += line_tokens
        turn_count += 1

    if current_lines:
        chunks.append({
            "content": "\n".join(current_lines),
            "timestamp_start": first_ts,
            "timestamp_end": last_ts,
            "turn_count": turn_count,
        })

    return chunks


def _real_import_timestamp(value) -> tuple[str, str] | None:
    """Return a real, parseable source timestamp; never invent a fallback."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return normalize_event_at(value.strip())
    except (TypeError, ValueError):
        return None


def _import_extraction_input(content: str) -> str:
    """Return the exact bounded, redacted text sent to the import model."""
    if not isinstance(content, str):
        raise ValueError("import content must be a string")
    return redact_embedding_input(content)[:12000]


def build_import_x_provenance(chunk: dict, chunk_index: int) -> dict:
    """Build truthful provenance for the exact import extraction input."""
    if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str):
        raise ValueError("import chunk content must be a string")
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")

    provenance = {
        "source_kind": "import",
        "source_digest": hashlib.sha256(
            _import_extraction_input(chunk["content"]).encode("utf-8")
        ).hexdigest(),
        "source_chunk_ordinal": chunk_index,
    }
    start = _real_import_timestamp(chunk.get("timestamp_start"))
    end = _real_import_timestamp(chunk.get("timestamp_end"))
    if start and end:
        try:
            start_dt = datetime.fromisoformat(start[0].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end[0].replace("Z", "+00:00"))
            compatible = (start_dt.tzinfo is None) == (end_dt.tzinfo is None)
            if compatible and start_dt <= end_dt:
                provenance["span_start"] = start[0]
                provenance["span_end"] = end[0]
        except (TypeError, ValueError):
            pass
    return provenance


# ============================================================
# Import State — persistent progress tracking
# 导入状态 — 持久化进度追踪
# ============================================================

class ImportState:
    """Manages import progress with file-based persistence."""

    def __init__(self, state_dir: str):
        self.state_file = os.path.join(state_dir, "import_state.json")
        self.lock_file = os.path.join(state_dir, ".import_state.lock")
        self.data = {
            "schema_version": IMPORT_STATE_SCHEMA_VERSION,
            "source_file": "",
            "source_hash": "",
            "total_chunks": 0,
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_merged": 0,
            "memories_raw": 0,
            "errors": [],
            "status": "idle",  # idle | running | paused | completed | error
            "started_at": "",
            "updated_at": "",
            "options": {},
            "chunks": [],
        }

    def load(self) -> bool:
        """Load state from file. Returns True if state exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                raise ImportStorageError(
                    "import ledger exists but cannot be read safely"
                ) from exc
            if not isinstance(saved, dict):
                raise ImportStorageError(
                    "import ledger root must be a JSON object"
                )
            self.data.update(saved)
            return True
        return False

    def save(self):
        """Persist state to file."""
        self.data["updated_at"] = now_iso()
        atomic_write_text(
            self.state_file,
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
        )

    def reset(
        self,
        source_file: str,
        source_hash: str,
        chunk_records: list[dict],
        *,
        preserve_raw: bool,
    ):
        """Reset state for a new import."""
        self.data = {
            "schema_version": IMPORT_STATE_SCHEMA_VERSION,
            "source_file": source_file,
            "source_hash": source_hash,
            "total_chunks": len(chunk_records),
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_merged": 0,
            "memories_raw": 0,
            "errors": [],
            "status": "running",
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "options": {"preserve_raw": bool(preserve_raw)},
            "chunks": chunk_records,
        }

    @property
    def can_resume(self) -> bool:
        return (
            self.data.get("status") in ("paused", "running", "error")
            and self.data.get("processed", 0) < self.data.get("total_chunks", 0)
        )

    def sync_summary(self) -> None:
        """Derive public counters from the durable output ledger."""
        chunks = self.data.get("chunks")
        if not isinstance(chunks, list):
            return

        completed = 0
        created = 0
        merged = 0
        raw = 0
        active_errors = []
        for chunk in chunks:
            if chunk.get("status") == "complete":
                completed += 1
            if chunk.get("status") == "error" and chunk.get("error"):
                active_errors.append(
                    f"Chunk {chunk.get('index', '?')}: {chunk['error'][:200]}"
                )
            for output in chunk.get("outputs", []):
                if output.get("status") != "complete":
                    continue
                action = output.get("action")
                if action == "merged":
                    merged += 1
                elif action == "raw":
                    raw += 1
                    created += 1
                elif action == "created":
                    created += 1

        self.data["processed"] = completed
        self.data["memories_created"] = created
        self.data["memories_merged"] = merged
        self.data["memories_raw"] = raw
        self.data["errors"] = active_errors[:100]

    def to_dict(self) -> dict:
        # Extraction payloads are required for crash-safe replay, but status
        # is exposed over HTTP and must not become a second memory-content API.
        public = json.loads(json.dumps(self.data, ensure_ascii=False))
        for chunk in public.get("chunks", []):
            for output in chunk.get("outputs", []):
                output.pop("item", None)
        return public


# ============================================================
# Import extraction prompt
# 导入提取提示词
# ============================================================

IMPORT_EXTRACT_PROMPT = """你是一个对话记忆提取专家。从以下对话片段中提取值得长期记住的信息。

提取规则：
1. 提取用户的事实、偏好、习惯、重要事件、情感时刻
2. 同一话题的零散信息整合为一条记忆
3. 过滤掉纯技术调试输出、代码块、重复问答、无意义寒暄
4. 如果对话中有特殊暗号、仪式性行为、关键承诺等，标记 preserve_raw=true
5. 如果内容是用户和AI之间的习惯性互动模式（例如打招呼方式、告别习惯），标记 is_pattern=true
6. 每条记忆不少于30字
7. 总条目数控制在 0~5 个（没有值得记的就返回空数组）
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记

输出格式（纯 JSON 数组，无其他内容）：
[
  {
    "name": "条目标题（10字以内）",
    "content": "整理后的内容",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1"],
    "importance": 5,
    "preserve_raw": false,
    "is_pattern": false
  }
]

主题域可选（选 1~2 个）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]

importance: 1-10
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）
preserve_raw: true = 特殊情境/暗号/仪式，保留原文不摘要
is_pattern: true = 反复出现的习惯性行为模式"""


# ============================================================
# Import Engine — core processing logic
# 导入引擎 — 核心处理逻辑
# ============================================================

class ImportEngine:
    """
    Processes conversation history files into OB memory buckets.
    将对话历史文件处理为 OB 记忆桶。
    """

    def __init__(self, config: dict, bucket_mgr, dehydrator, embedding_engine=None):
        self.config = config
        self.bucket_mgr = bucket_mgr
        self.dehydrator = dehydrator
        self.embedding_engine = embedding_engine
        # Optional best-effort callback installed by server.py.  Keeping this
        # out of the constructor preserves the existing ImportEngine API for
        # standalone users and tests.
        self.content_sync = None
        self.state = ImportState(config["buckets_dir"])
        self._maintenance_barrier = getattr(
            bucket_mgr,
            "_maintenance_barrier",
            None,
        ) or MaintenanceBarrier(config["buckets_dir"])
        self._paused = False
        self._running = False
        self._chunks: list[dict] = []

    @property
    def is_running(self) -> bool:
        return self._running

    def pause(self):
        """Request pause — will stop after current chunk finishes."""
        self._paused = True

    def get_status(self) -> dict:
        """Get current import status."""
        return self.state.to_dict()

    async def start(
        self,
        raw_content: str,
        filename: str = "",
        preserve_raw: bool = False,
        resume: bool = False,
    ) -> dict:
        """
        Start or resume an import.
        开始或恢复导入。
        """
        if self._running:
            return {"error": "Import already running"}

        self._running = True
        self._paused = False

        try:
            # A single durable ledger is shared by all workers using this
            # buckets_dir.  Serialize the complete import so two processes
            # cannot both observe a missing output marker and create twins.
            async with self._maintenance_barrier.shared_async():
                with advisory_file_lock(self.state.lock_file):
                    return await self._start_locked(
                        raw_content,
                        filename=filename,
                        preserve_raw=preserve_raw,
                        resume=resume,
                    )
        except Exception as e:
            try:
                async with self._maintenance_barrier.shared_async():
                    self.state.data["status"] = "error"
                    self.state.data["errors"] = [str(e)[:200]]
                    self.state.save()
            except Exception:
                logger.exception("Failed to persist terminal import error")
            raise
        finally:
            self._running = False

    async def _start_locked(
        self,
        raw_content: str,
        *,
        filename: str,
        preserve_raw: bool,
        resume: bool,
    ) -> dict:
        source_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        turns = detect_and_parse(raw_content, filename)
        if not turns:
            return {"error": "No conversation turns found in file"}

        self._chunks = chunk_turns(turns)
        if not self._chunks:
            return {"error": "No processable chunks after splitting"}

        if resume and self.state.load():
            saved_hash = str(self.state.data.get("source_hash", ""))
            hash_matches = saved_hash == source_hash
            # Read compatibility for the old truncated hash is allowed only
            # before any chunk was acknowledged.  Old progressed ledgers did
            # not record which failures were skipped and cannot be resumed
            # safely without manual reconciliation.
            if len(saved_hash) == 16 and source_hash.startswith(saved_hash):
                hash_matches = True
            if not hash_matches:
                raise RuntimeError("Source file changed; refusing unsafe resume")

            saved_options = self.state.data.get("options") or {}
            if (
                "preserve_raw" in saved_options
                and bool(saved_options["preserve_raw"]) != bool(preserve_raw)
            ):
                raise RuntimeError(
                    "preserve_raw changed; refusing to reinterpret a resumed import"
                )
            self._validate_or_initialize_resume_ledger(
                source_hash,
                preserve_raw=preserve_raw,
            )
            if self.state.data.get("status") == "completed":
                if all(
                    chunk.get("status") == "complete"
                    for chunk in self.state.data.get("chunks", [])
                ):
                    logger.info("Import is already durably complete")
                    return self.state.to_dict()
                raise RuntimeError(
                    "Import state says completed but its chunk ledger does not"
                )
            if not self.state.can_resume:
                raise RuntimeError(
                    f"Import in status {self.state.data.get('status')!r} "
                    "cannot be resumed"
                )
            logger.info(
                "Resuming import with %s/%s chunks durably complete",
                self.state.data["processed"],
                self.state.data["total_chunks"],
            )
            self.state.data["status"] = "running"
            self.state.save()
            return await self._process_chunks()

        records = [
            self._new_chunk_record(source_hash, i, chunk)
            for i, chunk in enumerate(self._chunks)
        ]
        self.state.reset(
            filename,
            source_hash,
            records,
            preserve_raw=preserve_raw,
        )
        self.state.save()

        logger.info(
            "Starting import: %s turns → %s chunks",
            len(turns),
            len(self._chunks),
        )
        return await self._process_chunks()

    def _new_chunk_record(
        self,
        source_hash: str,
        chunk_index: int,
        chunk: dict,
    ) -> dict:
        return {
            "index": chunk_index,
            "chunk_id": _chunk_identity(source_hash, chunk_index, chunk),
            "status": "pending",
            "attempts": 0,
            "extraction_status": "pending",
            "zero_candidates": False,
            "outputs": [],
            "error": "",
        }

    def _validate_or_initialize_resume_ledger(
        self,
        source_hash: str,
        *,
        preserve_raw: bool,
    ) -> None:
        records = self.state.data.get("chunks")
        if not isinstance(records, list):
            if self.state.data.get("processed", 0):
                raise RuntimeError(
                    "Legacy import progress has no per-chunk ledger; "
                    "manual reconciliation is required"
                )
            records = [
                self._new_chunk_record(source_hash, i, chunk)
                for i, chunk in enumerate(self._chunks)
            ]
            self.state.data["chunks"] = records

        if len(records) != len(self._chunks):
            raise RuntimeError("Chunk count changed; refusing unsafe resume")

        for i, (record, chunk) in enumerate(zip(records, self._chunks)):
            expected = _chunk_identity(source_hash, i, chunk)
            if record.get("index") != i or record.get("chunk_id") != expected:
                raise RuntimeError(
                    f"Chunk identity changed at index {i}; refusing unsafe resume"
                )
            if record.get("status") not in CHUNK_STATUSES:
                raise RuntimeError(f"Invalid chunk status at index {i}")
            for output in record.get("outputs", []):
                if output.get("status") not in OUTPUT_STATUSES:
                    raise RuntimeError(
                        f"Invalid output status in chunk {i}"
                    )
                if (
                    output.get("status") != "complete"
                    and not isinstance(output.get("item"), dict)
                ):
                    raise RuntimeError(
                        f"Missing replay payload in chunk {i}"
                    )

        self.state.data["schema_version"] = IMPORT_STATE_SCHEMA_VERSION
        self.state.data["source_hash"] = source_hash
        self.state.data["total_chunks"] = len(records)
        self.state.data["options"] = {"preserve_raw": bool(preserve_raw)}
        self.state.sync_summary()

    async def _process_chunks(self) -> dict:
        """Process every non-complete chunk without acknowledging failures."""
        for i, chunk in enumerate(self._chunks):
            record = self.state.data["chunks"][i]
            if record.get("status") == "complete":
                continue
            if self._paused:
                self.state.data["status"] = "paused"
                self.state.sync_summary()
                self.state.save()
                logger.info("Import paused at chunk %s/%s", i, len(self._chunks))
                return self.state.to_dict()

            try:
                await self._process_single_chunk(
                    chunk,
                    chunk_index=i,
                )
            except Exception as e:
                logger.warning("Import chunk error: Chunk %s: %s", i, str(e)[:200])

        self.state.sync_summary()
        statuses = {
            record.get("status")
            for record in self.state.data.get("chunks", [])
        }
        if "error" in statuses:
            self.state.data["status"] = "error"
        elif statuses <= {"complete"}:
            self.state.data["status"] = "completed"
        else:
            # pending/running/deferred can only remain after an explicit pause
            # or an interrupted worker; never label that batch completed.
            self.state.data["status"] = "paused"
        self.state.save()
        logger.info(
            "Import %s: %s/%s chunks complete, %s created, %s merged",
            self.state.data["status"],
            self.state.data["processed"],
            self.state.data["total_chunks"],
            self.state.data["memories_created"],
            self.state.data["memories_merged"],
        )
        return self.state.to_dict()

    async def _process_single_chunk(
        self,
        chunk: dict,
        *,
        chunk_index: int,
    ):
        """Extract and durably store one chunk using its replay ledger."""
        record = self.state.data["chunks"][chunk_index]
        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["error"] = ""
        self.state.sync_summary()
        self.state.save()

        content = chunk["content"]
        try:
            if record.get("extraction_status") != "complete":
                if content.strip():
                    try:
                        items = await self._extract_memories(content)
                    finally:
                        self.state.data["api_calls"] += 1
                else:
                    # An actually empty normalized chunk is a legitimate
                    # zero-candidate result, not a provider response.
                    items = []

                outputs = []
                preserve_all = bool(
                    self.state.data.get("options", {}).get("preserve_raw", False)
                )
                for output_index, item in enumerate(items):
                    should_preserve = preserve_all or bool(
                        item.get("preserve_raw", False)
                    )
                    output_id = _output_identity(
                        record["chunk_id"],
                        output_index,
                        item,
                        should_preserve,
                    )
                    outputs.append({
                        "output_id": output_id,
                        "status": "pending",
                        "attempts": 0,
                        "item": item,
                        "preserve_raw": should_preserve,
                        "requires_embedding": self._embedding_is_required(),
                        "bucket_id": "",
                        "action": "",
                        "error": "",
                    })

                # Persist the exact extraction result before the first bucket
                # mutation.  Retries never ask the model to reinterpret a
                # partially stored chunk.
                record["outputs"] = outputs
                record["zero_candidates"] = len(outputs) == 0
                record["extraction_status"] = "complete"
                self.state.save()

            for output in record.get("outputs", []):
                if output.get("status") == "complete":
                    continue
                output["status"] = "running"
                output["attempts"] = int(output.get("attempts", 0)) + 1
                output["error"] = ""
                self.state.save()
                try:
                    action, bucket_id = await self._store_output(
                        chunk,
                        chunk_index,
                        output,
                    )
                except Exception as e:
                    output["status"] = "error"
                    output["error"] = str(e)[:500]
                    raise
                output["status"] = "complete"
                output["action"] = action
                output["bucket_id"] = bucket_id
                output["error"] = ""
                # Completed outputs are recoverable from their durable bucket
                # marker and no longer need a second plaintext body in state.
                output.pop("item", None)
                self.state.sync_summary()
                self.state.save()

            record["status"] = "complete"
            record["error"] = ""
            self.state.sync_summary()
            self.state.save()
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)[:500]
            self.state.sync_summary()
            self.state.save()
            raise

    def _embedding_is_required(self) -> bool:
        return (
            self.embedding_engine is not None
            and bool(getattr(self.embedding_engine, "enabled", True))
        )

    async def _ensure_embedding(
        self,
        bucket_id: str,
        content: str,
        *,
        required: bool,
    ) -> None:
        if not required:
            return
        if self.embedding_engine is None:
            raise ImportStorageError(
                "required embedding engine is unavailable"
            )
        stored = await self.embedding_engine.generate_and_store(
            bucket_id,
            content,
        )
        if stored is not True:
            raise ImportStorageError(
                f"required embedding write failed for {bucket_id}"
            )

    async def _sync_written_content(self, bucket_id: str, content: str) -> None:
        """Notify additive sidecars after any durable content write/recovery."""
        callback = self.content_sync
        if not callable(callback):
            return
        try:
            result = callback(bucket_id, content)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            # Import durability must not depend on an additive index.  The
            # entity recall hash gate fails closed until a later recovery run.
            logger.warning(
                "Post-import content sync failed for %s: %s",
                bucket_id,
                type(exc).__name__,
            )

    async def _find_stored_output(
        self,
        output_id: str,
    ) -> tuple[dict, str] | None:
        list_all = getattr(self.bucket_mgr, "list_all", None)
        if not callable(list_all):
            raise ImportStorageError(
                "bucket manager cannot reconcile import output markers"
            )
        try:
            buckets = await list_all(include_archive=True)
        except Exception as e:
            raise ImportStorageError(
                f"cannot reconcile prior import outputs: {e}"
            ) from e

        prefix = _output_marker_prefix(output_id)
        matches = []
        for bucket in buckets:
            metadata = bucket.get("metadata") or {}
            tags = metadata.get("tags") or []
            markers = [
                tag for tag in tags
                if isinstance(tag, str) and tag.startswith(prefix)
            ]
            for marker in markers:
                action = marker[len(prefix):]
                if action in {"created", "merged", "raw"}:
                    matches.append((bucket, action))

        if len(matches) > 1:
            raise ImportStorageError(
                f"duplicate durable buckets for import output {output_id}"
            )
        return matches[0] if matches else None

    async def _store_output(
        self,
        chunk: dict,
        chunk_index: int,
        output: dict,
    ) -> tuple[str, str]:
        item = output["item"]
        output_id = output["output_id"]
        required_embedding = bool(output.get("requires_embedding", False))
        recovered = await self._find_stored_output(output_id)
        if recovered:
            bucket, action = recovered
            await self._sync_written_content(
                bucket["id"],
                bucket.get("content", item["content"]),
            )
            await self._ensure_embedding(
                bucket["id"],
                bucket.get("content", item["content"]),
                required=required_embedding,
            )
            return action, bucket["id"]

        x_provenance = build_import_x_provenance(chunk, chunk_index)
        source_time = _real_import_timestamp(chunk.get("timestamp_start"))
        time_kwargs = {}
        if source_time:
            time_kwargs = {
                "event_at": source_time[0],
                "date_precision": source_time[1],
                "date_source": "import_source_timestamp",
                "date_confidence": 1.0,
            }

        if output.get("preserve_raw"):
            marker = _output_marker(output_id, "raw")
            bucket_id = await self.bucket_mgr.create(
                content=item["content"],
                tags=_dedupe_tags(item.get("tags", []) + [marker]),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name"),
                x_provenance=x_provenance,
                **time_kwargs,
            )
            if not bucket_id:
                raise ImportStorageError("bucket create returned no id")
            await self._sync_written_content(bucket_id, item["content"])
            await self._ensure_embedding(
                bucket_id,
                item["content"],
                required=required_embedding,
            )
            return "raw", bucket_id

        return await self._merge_or_create_item_result(
            item,
            x_provenance=x_provenance,
            time_kwargs=time_kwargs,
            output_id=output_id,
            requires_embedding=required_embedding,
        )

    async def _extract_memories(self, chunk_content: str) -> list[dict]:
        """Use LLM to extract memories from a conversation chunk."""
        if not self.dehydrator.api_available:
            raise RuntimeError("API not available")

        safe_chunk = _import_extraction_input(chunk_content)
        response = await self.dehydrator.client.chat.completions.create(
            model=self.dehydrator.model,
            messages=[
                {"role": "system", "content": IMPORT_EXTRACT_PROMPT},
                {"role": "user", "content": safe_chunk},
            ],
            max_tokens=2048,
            temperature=0.0,
        )

        if not getattr(response, "choices", None):
            raise ImportExtractionError("provider returned no choices")
        message = getattr(response.choices[0], "message", None)
        if message is None:
            raise ImportExtractionError("provider returned no message")
        raw = getattr(message, "content", None) or ""
        if not raw.strip():
            raise ImportExtractionError("provider returned an empty body")

        return self._parse_extraction(raw)

    @staticmethod
    def _parse_extraction(raw: str) -> list[dict]:
        """Parse and validate LLM extraction result."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            logger.warning(f"Import extraction JSON parse failed: {raw[:200]}")
            raise ImportExtractionError("provider returned invalid JSON") from e

        if not isinstance(items, list):
            raise ImportExtractionError("provider response must be a JSON array")
        if len(items) > 5:
            raise ImportExtractionError("provider returned more than 5 memories")

        validated = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ImportExtractionError(
                    f"memory {index} is not a JSON object"
                )
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ImportExtractionError(
                    f"memory {index} has no non-empty content"
                )

            domain = item.get("domain", ["未分类"])
            tags = item.get("tags", [])
            if not isinstance(domain, list) or not all(
                isinstance(value, str) and value.strip() for value in domain
            ):
                raise ImportExtractionError(
                    f"memory {index} has invalid domain"
                )
            if not isinstance(tags, list) or not all(
                isinstance(value, str) for value in tags
            ):
                raise ImportExtractionError(
                    f"memory {index} has invalid tags"
                )
            for bool_field in ("preserve_raw", "is_pattern"):
                if (
                    bool_field in item
                    and type(item[bool_field]) is not bool
                ):
                    raise ImportExtractionError(
                        f"memory {index} has invalid {bool_field}"
                    )

            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "content": content,
                "domain": domain[:3] or ["未分类"],
                "valence": valence,
                "arousal": arousal,
                "tags": tags[:10],
                "importance": importance,
                "preserve_raw": bool(item.get("preserve_raw", False)),
                "is_pattern": bool(item.get("is_pattern", False)),
            })

        return validated

    async def _merge_or_create_item(
        self,
        item: dict,
        *,
        x_provenance: dict | None = None,
        time_kwargs: dict | None = None,
    ) -> bool:
        """Try to merge with existing bucket, or create new. Returns is_merged."""
        action, _bucket_id = await self._merge_or_create_item_result(
            item,
            x_provenance=x_provenance,
            time_kwargs=time_kwargs,
            output_id=None,
            requires_embedding=self._embedding_is_required(),
        )
        return action == "merged"

    async def _merge_or_create_item_result(
        self,
        item: dict,
        *,
        x_provenance: dict | None = None,
        time_kwargs: dict | None = None,
        output_id: str | None,
        requires_embedding: bool,
    ) -> tuple[str, str]:
        """Merge/create and return the durable action and bucket id."""
        content = item["content"]
        domain = item.get("domain", ["未分类"])
        tags = item.get("tags", [])
        importance = item.get("importance", 5)
        valence = item.get("valence", 0.5)
        arousal = item.get("arousal", 0.3)
        name = item.get("name", "")

        try:
            existing = await self.bucket_mgr.search(content, limit=1, domain_filter=domain or None)
        except Exception as e:
            if output_id:
                raise ImportStorageError(
                    f"bucket search failed before import write: {e}"
                ) from e
            existing = []

        merge_threshold = self.config.get("merge_threshold", 75)

        if existing and existing[0].get("score", 0) > merge_threshold:
            bucket = existing[0]
            if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
                merged = None
                try:
                    try:
                        merged = await self.dehydrator.merge(
                            bucket["content"],
                            content,
                        )
                    finally:
                        self.state.data["api_calls"] += 1
                    if not isinstance(merged, str) or not merged.strip():
                        raise ImportExtractionError(
                            "merge provider returned an empty body"
                        )
                    old_v = bucket["metadata"].get("valence", 0.5)
                    old_a = bucket["metadata"].get("arousal", 0.3)
                    merge_tags = bucket["metadata"].get("tags", []) + tags
                    if output_id:
                        merge_tags.append(
                            _output_marker(output_id, "merged")
                        )
                    updated = await self.bucket_mgr.update(
                        bucket["id"],
                        content=merged,
                        tags=_dedupe_tags(merge_tags),
                        importance=max(bucket["metadata"].get("importance", 5), importance),
                        domain=_dedupe_tags(
                            bucket["metadata"].get("domain", []) + domain
                        ),
                        valence=round((old_v + valence) / 2, 2),
                        arousal=round((old_a + arousal) / 2, 2),
                    )
                    if updated is not True:
                        raise ImportStorageError(
                            f"bucket update failed for {bucket['id']}"
                        )
                    await self._sync_written_content(bucket["id"], merged)
                    await self._ensure_embedding(
                        bucket["id"],
                        merged,
                        required=requires_embedding,
                    )
                    return "merged", bucket["id"]
                except Exception as e:
                    # ``BucketManager.update`` may fail after its atomic file
                    # replacement.  Re-read synchronization is safe whether
                    # the old or new body is durable and prevents stale links
                    # while the import ledger waits for recovery.
                    await self._sync_written_content(
                        bucket["id"],
                        merged if isinstance(merged, str) else bucket["content"],
                    )
                    logger.warning(f"Merge failed during import: {e}")
                    if output_id:
                        raise

        # Create new
        create_tags = list(tags)
        if output_id:
            create_tags.append(_output_marker(output_id, "created"))
        bucket_id = await self.bucket_mgr.create(
            content=content,
            tags=_dedupe_tags(create_tags),
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name or None,
            x_provenance=x_provenance,
            **(time_kwargs or {}),
        )
        if not bucket_id:
            raise ImportStorageError("bucket create returned no id")
        await self._sync_written_content(bucket_id, content)
        await self._ensure_embedding(
            bucket_id,
            content,
            required=requires_embedding,
        )
        return "created", bucket_id

    async def detect_patterns(self) -> list[dict]:
        """
        Post-import: detect high-frequency patterns via embedding clustering.
        导入后：通过 embedding 聚类检测高频模式。
        Returns list of {pattern_content, count, bucket_ids, suggested_action}.
        """
        if not self.embedding_engine:
            return []

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        dynamic_buckets = [
            b for b in all_buckets
            if b["metadata"].get("type") == "dynamic"
            and not b["metadata"].get("pinned")
            and not b["metadata"].get("resolved")
        ]

        if len(dynamic_buckets) < 5:
            return []

        # Get embeddings
        embeddings = {}
        for b in dynamic_buckets:
            emb = await self.embedding_engine.get_embedding(b["id"])
            if emb is not None:
                embeddings[b["id"]] = emb

        if len(embeddings) < 5:
            return []

        # Find clusters: group by pairwise similarity > 0.7
        import numpy as np
        ids = list(embeddings.keys())
        clusters: dict[str, list[str]] = {}
        visited = set()

        for i, id_a in enumerate(ids):
            if id_a in visited:
                continue
            cluster = [id_a]
            visited.add(id_a)
            emb_a = np.array(embeddings[id_a])
            norm_a = np.linalg.norm(emb_a)
            if norm_a == 0:
                continue

            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in visited:
                    continue
                emb_b = np.array(embeddings[id_b])
                norm_b = np.linalg.norm(emb_b)
                if norm_b == 0:
                    continue
                sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                if sim > 0.7:
                    cluster.append(id_b)
                    visited.add(id_b)

            if len(cluster) >= 3:
                clusters[id_a] = cluster

        # Format results
        patterns = []
        for lead_id, cluster_ids in clusters.items():
            lead_bucket = next((b for b in dynamic_buckets if b["id"] == lead_id), None)
            if not lead_bucket:
                continue
            patterns.append({
                "pattern_content": redact_text(lead_bucket["content"])[:200],
                "pattern_name": lead_bucket["metadata"].get("name", lead_id),
                "count": len(cluster_ids),
                "bucket_ids": cluster_ids,
                "suggested_action": "pin" if len(cluster_ids) >= 5 else "review",
            })

        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns[:20]
