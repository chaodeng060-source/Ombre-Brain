# ============================================================
# Shared test fixtures — isolated temp environment for all tests
# 共享测试 fixtures —— 为所有测试提供隔离的临时环境
#
# IMPORTANT: All tests run against a temp directory.
# Your real /data or local buckets are NEVER touched.
# 重要：所有测试在临时目录运行，绝不触碰真实记忆数据。
# ============================================================

import os
import sys
import math
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def test_config(tmp_path):
    """Minimal config pointing to a temp directory."""
    buckets_dir = str(tmp_path / "buckets")
    os.makedirs(os.path.join(buckets_dir, "permanent"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "dynamic"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "archive"), exist_ok=True)
    os.makedirs(os.path.join(buckets_dir, "dynamic", "feel"), exist_ok=True)

    return {
        "buckets_dir": buckets_dir,
        "audit": {"enabled": False},
        "matching": {"fuzzy_threshold": 50, "max_results": 10},
        "wikilink": {"enabled": False},
        "scoring_weights": {
            "topic_relevance": 4.0,
            "emotion_resonance": 2.0,
            "time_proximity": 2.5,
            "importance": 1.0,
            "content_weight": 3.0,
        },
        "decay": {
            "lambda": 0.05,
            "threshold": 0.3,
            "check_interval_hours": 24,
            "emotion_weights": {"base": 1.0, "arousal_boost": 0.8},
        },
        "dehydration": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash-lite",
        },
        "embedding": {
            "api_key": os.environ.get("OMBRE_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-embedding-001",
        },
    }


@pytest.fixture
def bucket_mgr(test_config):
    from bucket_manager import BucketManager
    return BucketManager(test_config)


@pytest.fixture
def decay_eng(test_config, bucket_mgr):
    from decay_engine import DecayEngine
    return DecayEngine(test_config, bucket_mgr)
