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
from datetime import datetime


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
        "matching": {
            "fuzzy_threshold": 50,
            "max_results": 5,
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
