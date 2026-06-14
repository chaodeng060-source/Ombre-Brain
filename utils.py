# ============================================================
# Module: Common Utilities (utils.py)
# 模块：通用工具函数
#
# Provides config loading, logging init, path safety, ID generation, etc.
# 提供配置加载、日志初始化、路径安全校验、ID 生成等基础能力
#
# Depended on by: server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# 被谁依赖：server.py, bucket_manager.py, dehydrator.py, decay_engine.py
# ============================================================

import os
import re
import uuid
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta


# 6 类关系边：causes/contributes/improves/explains/updates 有向，kin 无向（仍单边记一次）
RELATION_TYPES = frozenset({
    "causes",       # 触发/导致
    "contributes",  # 贡献
    "improves",     # 改善
    "explains",     # 解释
    "updates",      # 更新（A 取代/补正 B）
    "kin",          # 同类
})

# Domains where a bucket must never be marked resolved=True.
# Not "problem-and-solution" structures — persistent states (relationships,
# commitments, feelings, family, self-reflection). Resolving = forgetting.
# 5.10 incident: a CC self over-zealously resolved 13 buckets in this set;
# code-level guard supersedes "trust the bot to read the iron rule".
PROTECTED_RESOLVE_DOMAINS = frozenset({
    "恋爱", "纪念日", "约定", "家庭", "自省", "feel",
})


class ResolvedGuardError(Exception):
    """Refused setting resolved=True on a protected-domain bucket."""
    pass


def load_config(config_path: str = None) -> dict:
    """
    Load configuration file.
    加载配置文件。

    Priority: environment variables > config.yaml > built-in defaults.
    优先级：环境变量 > config.yaml > 内置默认值。
    """
    defaults = {
        "transport": "stdio",
        "log_level": "INFO",
        "buckets_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckets"),
        "merge_threshold": 75,
        "dehydration": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "",
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        "embedding": {
            "enabled": True,
            "model": "gemini-embedding-001",
            "base_url": "",
            "api_key": "",
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {
                "base": 1.0,
                "arousal_boost": 0.8,
            },
        },
        "briefing": {
            # 简报「最近活跃」叙事段的绝对年龄闸（按 created 计，单位天）。
            # last_active 会被 inspect/backfill_relations/touch 等维护操作 bump，
            # 旧桶会冒充「最近活跃」被 LLM 写成「前两天」。created 早于此窗的桶
            # 只许走 pinned/protected/未解决权重池，不许进「上一窗口/再之前」叙事。
            # (2026-06-08 修：朝灯戳穿一个月前的卡兜事被当成前两天)
            "recent_max_age_days": 7,
            "protected_verbatim_limit": 6,
        },
        # --- 五感入口层 / Sense entry layer ---
        # 写入时给带感官描述的内容打 sense 标签，breath 时同感官 query 让相关桶上浮（普鲁斯特钩子）。
        # 代码各处已有同值 fallback，这里补进 defaults：让配置可见、可热改、可被 env 覆盖（小卷 #5）。
        "sense": {
            "enabled": True,
            "recall_boost": 1.25,
        },
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
        },
        "merge": {
            "keyword_limit": 5,
            "vector_limit": 8,
            "vector_floor": 0.50,
            "vector_threshold": 0.78,
            "candidate_limit": 8,
        },
        # --- World axis / 世界轴 ---
        # worlds: 可选值清单，trace/hold 接受的合法 world。空字符串 "" 表示日常聊天（不属于任何世界）。
        # current_world: 全局指针。hold 不传 world 时走这个；breath 不传 world 时按这个过滤。
        # 角色扮演时切到具体世界，breath 默认只出"该世界 + 通用"，日常桶不浮现。
        # current_world="" 即日常模式：breath 默认出"日常桶 + 通用"，角色世界桶不浮现。
        "worlds": ["当前世界", "旧世界", "通用"],
        "current_world": "",
    }

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )

    config = defaults.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            if isinstance(file_config, dict):
                config = _deep_merge(defaults, file_config)
            else:
                logging.warning(
                    f"Config file is not a valid YAML dict, using defaults / "
                    f"配置文件不是有效的 YAML 字典，使用默认配置: {config_path}"
                )
        except yaml.YAMLError as e:
            logging.warning(
                f"Failed to parse config file, using defaults / "
                f"配置文件解析失败，使用默认配置: {e}"
            )

    # --- Dehydration env overrides ---
    env_api_key = os.environ.get("OMBRE_API_KEY", "")
    if env_api_key:
        config.setdefault("dehydration", {})["api_key"] = env_api_key

    env_base_url = os.environ.get("OMBRE_BASE_URL", "")
    if env_base_url:
        config.setdefault("dehydration", {})["base_url"] = env_base_url

    env_model = os.environ.get("OMBRE_MODEL", "")
    if env_model:
        config.setdefault("dehydration", {})["model"] = env_model

    # --- Embedding env overrides (independent from dehydration) ---
    env_embed_api_key = os.environ.get("OMBRE_EMBED_API_KEY", "")
    if env_embed_api_key:
        config.setdefault("embedding", {})["api_key"] = env_embed_api_key

    env_embed_base_url = os.environ.get("OMBRE_EMBED_BASE_URL", "")
    if env_embed_base_url:
        config.setdefault("embedding", {})["base_url"] = env_embed_base_url

    env_embed_model = os.environ.get("OMBRE_EMBED_MODEL", "")
    if env_embed_model:
        config.setdefault("embedding", {})["model"] = env_embed_model

    # --- Matching env overrides ---
    env_fuzzy = os.environ.get("OMBRE_FUZZY_THRESHOLD", "")
    if env_fuzzy:
        try:
            config.setdefault("matching", {})["fuzzy_threshold"] = int(env_fuzzy)
        except ValueError:
            pass

    # --- Sense env overrides ---
    env_sense_enabled = os.environ.get("OMBRE_SENSE_ENABLED", "")
    if env_sense_enabled:
        config.setdefault("sense", {})["enabled"] = (
            env_sense_enabled.lower() not in ("0", "false", "no", "off")
        )

    env_sense_boost = os.environ.get("OMBRE_SENSE_RECALL_BOOST", "")
    if env_sense_boost:
        try:
            config.setdefault("sense", {})["recall_boost"] = float(env_sense_boost)
        except ValueError:
            pass

    # --- Misc env overrides ---
    env_transport = os.environ.get("OMBRE_TRANSPORT", "")
    if env_transport:
        config["transport"] = env_transport

    env_buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "")
    if env_buckets_dir:
        config["buckets_dir"] = env_buckets_dir

    env_current_world = os.environ.get("OMBRE_CURRENT_WORLD", "")
    if env_current_world:
        config["current_world"] = env_current_world

    # --- Ensure bucket storage directories exist ---
    buckets_dir = config["buckets_dir"]
    for subdir in ["permanent", "dynamic", "archive"]:
        os.makedirs(os.path.join(buckets_dir, subdir), exist_ok=True)

    # --- Runtime state sidecar overrides current_world ---
    # runtime 状态文件在 buckets 目录下，switch_world MCP 工具会写它。
    # 优先级：env > sidecar > config.yaml > defaults。env 覆盖在前面已应用。
    if not env_current_world:
        sidecar_world = _load_runtime_current_world(buckets_dir)
        if sidecar_world is not None:
            config["current_world"] = sidecar_world

    return config


def _runtime_state_path(buckets_dir: str) -> str:
    return os.path.join(buckets_dir, ".ombre_runtime.yaml")


def _load_runtime_current_world(buckets_dir: str):
    path = _runtime_state_path(buckets_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and "current_world" in data:
            return str(data.get("current_world") or "")
    except (yaml.YAMLError, OSError):
        return None
    return None


def save_current_world(buckets_dir: str, value: str) -> None:
    """Persist current_world to runtime sidecar. switch_world MCP tool calls this."""
    path = _runtime_state_path(buckets_dir)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                data = {}
        except (yaml.YAMLError, OSError):
            data = {}
    data["current_world"] = str(value or "")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


def generate_bucket_id() -> str:
    return uuid.uuid4().hex[:12]


def strip_wikilinks(text: str) -> str:
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text) if text else text


def sanitize_name(name: str) -> str:
    if not isinstance(name, str):
        return "unnamed"
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", name, flags=re.UNICODE)
    cleaned = cleaned.strip()[:80]
    return cleaned if cleaned else "unnamed"


def safe_path(base_dir: str, filename: str) -> Path:
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(
            f"Path safety check failed / 路径安全检查失败: "
            f"{target} is not inside / 不在 {base} 内"
        )
    return target


def count_tokens_approx(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.05)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Time string parsing / 时间字符串解析 ---
# Used by breath() since/until params and bucket_manager.search created range.
# 给 breath since/until 和 bucket_manager.search created 范围过滤用。
#
# Supports three forms / 支持三种格式：
#   - ISO 8601: "2026-05-01" / "2026-05-01T12:00:00"
#   - Keywords: "now" / "today" (today 00:00) / "yesterday" (yesterday 00:00)
#   - Relative offsets: "-7d" / "-3h" / "-30m" / "+1d" / "7d" (positive default)
#
# Natural language like "上周/三天前" intentionally NOT supported here.
# Caller (LLM/Claude) translates natural language to one of the above forms.
# 不做"上周/三天前"类中文 NLU，调用方（LLM）负责把人话翻成上面三种确定格式。

_RELATIVE_TIME_RE = re.compile(r"([+-]?)(\d+)([dhm])")


def parse_relative_time(s: str, reference: datetime = None):
    """
    Parse a time string into a datetime. Returns None on failure.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    ref = reference if reference is not None else datetime.now()

    low = s.lower()
    if low == "now":
        return ref
    if low == "today":
        return ref.replace(hour=0, minute=0, second=0, microsecond=0)
    if low == "yesterday":
        return (ref - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    m = _RELATIVE_TIME_RE.fullmatch(low)
    if m:
        sign, num, unit = m.groups()
        try:
            n = int(num)
        except ValueError:
            return None
        if sign == "-":
            n = -n
        if unit == "d":
            return ref + timedelta(days=n)
        if unit == "h":
            return ref + timedelta(hours=n)
        if unit == "m":
            return ref + timedelta(minutes=n)

    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


# --- World filter helpers / 世界轴过滤辅助 ---
# 桶 world="通用" 永远跟着任何 world filter 一起出。
UNIVERSAL_WORLD = "通用"


def world_matches(bucket_world: str, world_filter_set: set) -> bool:
    """判定一个桶的 world 是否通过 filter。
    world_filter_set 约定：
      - {""}                  日常模式：只让 world="" 桶 + world="通用" 通过
      - {"当前世界"}          单世界：只让 world="当前世界" + world="通用" 通过
      - {"当前世界","旧世界"} 多世界：两个 + 通用
      - 调用方需自行处理 None（None=不过滤，不应进入此函数）
    """
    bw = (bucket_world or "").strip()
    if bw == UNIVERSAL_WORLD:
        return True
    return bw in world_filter_set


# --- Reciprocal Rank Fusion / 倒数排名融合 ---
# Combines two ranked channels (keyword + vector) into a single fused ranking.
# 将两个 ranked 通道（关键词 + 向量）融合成一个统一排序。
#
# Formula: score(d) = w_k/(k + rank_k(d)) + w_v/(k + rank_v(d))
# 公式：桶 d 的融合分 = 关键词权重/(k + 关键词通道排名) + 向量权重/(k + 向量通道排名)
#
# Buckets present in only one channel only contribute that channel's term.
# 只在单通道出现的桶，只贡献该通道的项。
#
# k 默认 60（标准 RRF 取值）。k 越大，相邻 rank 间分差越平缓。

def rrf_fuse(
    keyword_ranked: list,
    vector_ranked: list,
    k: int = 60,
    keyword_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> list:
    """
    Reciprocal Rank Fusion of two ranked channels.

    Args:
        keyword_ranked: list of (bucket_id, _score_unused), top-down ordered
        vector_ranked:  list of (bucket_id, _score_unused), top-down ordered
        k:              rank smoothing constant (default 60)
        keyword_weight: weight for keyword channel (default 1.0)
        vector_weight:  weight for vector channel (default 1.0)

    Returns:
        list of (bucket_id, fused_score), sorted by fused_score descending
    """
    scores: dict = {}
    for rank, item in enumerate(keyword_ranked, start=1):
        bid = item[0]
        scores[bid] = scores.get(bid, 0.0) + keyword_weight / (k + rank)
    for rank, item in enumerate(vector_ranked, start=1):
        bid = item[0]
        scores[bid] = scores.get(bid, 0.0) + vector_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
