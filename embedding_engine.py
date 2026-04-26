# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via OpenAI-compatible API,
# stores them in SQLite, and provides cosine similarity search.
#
# Depended on by: server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存储 + 余弦搜索。

    Priority for credentials:
    1. embedding.* config (independent for embedding)
    2. dehydration.* config (fallback for backward compatibility)
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        # Independent credentials with fallback to dehydration config
        self.api_key = embed_cfg.get("api_key", "") or dehy_cfg.get("api_key", "")
        self.base_url = (
            embed_cfg.get("base_url", "")
            or dehy_cfg.get("base_url", "")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)

        # SQLite path
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content)
            if not embedding:
                return False
            self._store_embedding(bucket_id, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def _generate_embedding(self, text: str) -> list[float]:
        truncated = text[:2000]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (bucket_id, embedding, updated_at) VALUES (?, ?, ?)",
            (bucket_id, json.dumps(embedding), now_iso()),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.enabled:
            return []

        try:
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT bucket_id, embedding FROM embeddings").fetchall()
        conn.close()

        if not rows:
            return []

        results = []
        for bucket_id, emb_json in rows:
            try:
                stored_embedding = json.loads(emb_json)
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                results.append((bucket_id, sim))
            except (json.JSONDecodeError, Exception):
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
