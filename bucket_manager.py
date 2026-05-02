# ============================================================
# Module: Memory Bucket Manager (bucket_manager.py)
# 模块：记忆桶管理器
#
# CRUD operations, multi-dimensional index search, activation updates
# for memory buckets.
# 记忆桶的增删改查、多维索引搜索、激活更新。
#
# Core design:
# 核心逻辑：
#   - Each bucket = one Markdown file (YAML frontmatter + body)
#     每个记忆桶 = 一个 Markdown 文件
#   - Storage by type: permanent / dynamic / archive
#     存储按类型分目录
#   - Multi-dimensional soft index: domain + valence/arousal + fuzzy text
#     多维软索引：主题域 + 情感坐标 + 文本模糊匹配
#   - Search strategy: domain pre-filter → weighted multi-dim ranking
#     搜索策略：主题域预筛 → 多维加权精排
#   - Emotion coordinates based on Russell circumplex model:
#     情感坐标基于环形情感模型（Russell circumplex）：
#       valence (0~1): 0=negative → 1=positive
#       arousal (0~1): 0=calm → 1=excited
#
# Depended on by: server.py, decay_engine.py
# 被谁依赖：server.py, decay_engine.py
# ============================================================

import os
import math
import logging
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import frontmatter
import jieba
from rapidfuzz import fuzz

from utils import generate_bucket_id, sanitize_name, safe_path, now_iso

logger = logging.getLogger("ombre_brain.bucket")


class BucketManager:
    """
    Memory bucket manager — entry point for all bucket CRUD operations.
    Buckets are stored as Markdown files with YAML frontmatter for metadata
    and body for content. Natively compatible with Obsidian browsing/editing.
    记忆桶管理器 —— 所有桶的 CRUD 操作入口。
    桶以 Markdown 文件存储，YAML frontmatter 存元数据，正文存内容。
    天然兼容 Obsidian 直接浏览和编辑。
    """

    def __init__(self, config: dict):
        # --- Read storage paths from config / 从配置中读取存储路径 ---
        self.base_dir = config["buckets_dir"]
        self.permanent_dir = os.path.join(self.base_dir, "permanent")
        self.dynamic_dir = os.path.join(self.base_dir, "dynamic")
        self.archive_dir = os.path.join(self.base_dir, "archive")
        self.feel_dir = os.path.join(self.base_dir, "feel")
        self.fuzzy_threshold = config.get("matching", {}).get("fuzzy_threshold", 50)
        self.max_results = config.get("matching", {}).get("max_results", 5)

        # --- Wikilink config / 双链配置 ---
        wikilink_cfg = config.get("wikilink", {})
        self.wikilink_enabled = wikilink_cfg.get("enabled", True)
        self.wikilink_use_tags = wikilink_cfg.get("use_tags", False)
        self.wikilink_use_domain = wikilink_cfg.get("use_domain", True)
        self.wikilink_use_auto_keywords = wikilink_cfg.get("use_auto_keywords", True)
        self.wikilink_auto_top_k = wikilink_cfg.get("auto_top_k", 8)
        self.wikilink_min_len = wikilink_cfg.get("min_keyword_len", 2)
        self.wikilink_exclude_keywords = set(wikilink_cfg.get("exclude_keywords", []))
        self.wikilink_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "我们", "你们", "他们", "然后", "今天", "昨天", "明天", "一下",
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "with",
            "this", "that", "from", "they", "been", "said", "will", "each",
        }
        self.wikilink_stopwords |= {w.lower() for w in self.wikilink_exclude_keywords}

        # --- Search scoring weights / 检索权重配置 ---
        scoring = config.get("scoring_weights", {})
        self.w_topic = scoring.get("topic_relevance", 4.0)
        self.w_emotion = scoring.get("emotion_resonance", 2.0)
        self.w_time = scoring.get("time_proximity", 2.5)
        self.w_importance = scoring.get("importance", 1.0)
        self.content_weight = scoring.get("content_weight", 3.0)

    # ---------------------------------------------------------
    # Create a new bucket
    # 创建新桶
    # ---------------------------------------------------------
    async def create(
        self,
        content: str,
        tags: list[str] = None,
        importance: int = 5,
        domain: list[str] = None,
        valence: float = 0.5,
        arousal: float = 0.3,
        bucket_type: str = "dynamic",
        name: str = None,
        pinned: bool = False,
        protected: bool = False,
    ) -> str:
        bucket_id = generate_bucket_id()
        bucket_name = sanitize_name(name) if name else bucket_id
        domain = domain or ["未分类"]
        tags = tags or []
        linked_content = content

        if pinned or protected:
            importance = 10

        metadata = {
            "id": bucket_id,
            "name": bucket_name,
            "tags": tags,
            "domain": domain,
            "valence": max(0.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "importance": max(1, min(10, importance)),
            "type": bucket_type,
            "created": now_iso(),
            "last_active": now_iso(),
            "activation_count": 1,
        }
        if pinned:
            metadata["pinned"] = True
        if protected:
            metadata["protected"] = True

        # Defensive: ensure no 'content' key sneaks into metadata kwargs
        # 防御性：确保 metadata 里没有 content 键，否则会和 body 撞 Post() 参数
        metadata.pop("content", None)

        post = frontmatter.Post(linked_content, **metadata)

        if bucket_type == "permanent" or pinned:
            type_dir = self.permanent_dir
            if pinned and bucket_type != "permanent":
                metadata["type"] = "permanent"
        elif bucket_type == "feel":
            type_dir = self.feel_dir
        else:
            type_dir = self.dynamic_dir
            
        if bucket_type == "feel":
            primary_domain = "沉淀物"
        else:
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            
        target_dir = os.path.join(type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)

        if bucket_name and bucket_name != bucket_id:
            filename = f"{bucket_name}_{bucket_id}.md"
        else:
            filename = f"{bucket_id}.md"
        file_path = safe_path(target_dir, filename)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))
        except OSError as e:
            logger.error(f"Failed to write bucket file / 写入桶文件失败: {file_path}: {e}")
            raise

        logger.info(
            f"Created bucket / 创建记忆桶: {bucket_id} ({bucket_name}) → {primary_domain}/"
            + (" [PINNED]" if pinned else "") + (" [PROTECTED]" if protected else "")
        )
        return bucket_id

    # ---------------------------------------------------------
    # Read bucket content
    # ---------------------------------------------------------
    async def get(self, bucket_id: str) -> Optional[dict]:
        if not bucket_id or not isinstance(bucket_id, str):
            return None
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return None
        return self._load_bucket(file_path)

    def _move_bucket(self, file_path: str, target_type_dir: str, domain: list[str] = None) -> str:
        primary_domain = sanitize_name(domain[0]) if domain else "未分类"
        target_dir = os.path.join(target_type_dir, primary_domain)
        os.makedirs(target_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        new_path = safe_path(target_dir, filename)
        if os.path.normpath(file_path) != os.path.normpath(new_path):
            os.rename(file_path, new_path)
            logger.info(f"Moved bucket / 移动记忆桶: {filename} → {target_dir}/")
        return new_path

    # ---------------------------------------------------------
    # Update bucket
    # ---------------------------------------------------------
    async def update(self, bucket_id: str, **kwargs) -> bool:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = self._safe_load_post(file_path)
            old_domain = post.get("domain", ["未分类"])
            old_type = post.get("type", "dynamic")
            old_pinned = post.get("pinned", False)

            for key, value in kwargs.items():
                if value is not None:
                    if key == "content":
                        # 'content' is body, not metadata — avoid Post() collision
                        # 'content' 是正文不是元数据 — 防止 Post() 撞键
                        post.content = value
                    else:
                        post[key] = value

            post["last_active"] = now_iso()

            new_pinned = post.get("pinned", False)
            new_type = post.get("type", "dynamic")
            new_domain = post.get("domain", ["未分类"])

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            need_move = False
            target_type_dir = None

            if new_pinned and not old_pinned:
                target_type_dir = self.permanent_dir
                need_move = True
            elif not new_pinned and old_pinned:
                target_type_dir = self.dynamic_dir if new_type != "feel" else self.feel_dir
                need_move = True
            elif new_type != old_type:
                if new_type == "permanent":
                    target_type_dir = self.permanent_dir
                elif new_type == "feel":
                    target_type_dir = self.feel_dir
                else:
                    target_type_dir = self.dynamic_dir
                need_move = True
            elif new_domain != old_domain:
                if new_pinned or new_type == "permanent":
                    target_type_dir = self.permanent_dir
                elif new_type == "feel":
                    target_type_dir = self.feel_dir
                else:
                    target_type_dir = self.dynamic_dir
                need_move = True

            if need_move and target_type_dir:
                self._move_bucket(file_path, target_type_dir, new_domain)

        except Exception as e:
            logger.error(f"Failed to update bucket / 更新桶失败: {bucket_id}: {e}")
            return False

        logger.info(f"Updated bucket / 更新记忆桶: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Delete bucket
    # ---------------------------------------------------------
    async def delete(self, bucket_id: str) -> bool:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = self._safe_load_post(file_path)
            if post.get("protected", False):
                logger.warning(f"Cannot delete protected bucket / 受保护的桶不可删除: {bucket_id}")
                return False
        except Exception as e:
            logger.warning(f"Failed to check protection on {bucket_id}: {e}")

        try:
            os.remove(file_path)
        except OSError as e:
            logger.error(f"Failed to delete bucket file / 删除桶文件失败: {bucket_id}: {e}")
            return False

        logger.info(f"Deleted bucket / 删除记忆桶: {bucket_id}")
        return True

    # ---------------------------------------------------------
    # Touch bucket
    # ---------------------------------------------------------
    async def touch(self, bucket_id: str) -> None:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return

        try:
            post = self._safe_load_post(file_path)
            post["last_active"] = now_iso()
            post["activation_count"] = post.get("activation_count", 0) + 1

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            current_time = datetime.fromisoformat(str(post.get("created", post.get("last_active", ""))))
            await self._time_ripple(bucket_id, current_time)
        except Exception as e:
            logger.warning(f"Failed to touch bucket / 触碰桶失败: {bucket_id}: {e}")

    async def _time_ripple(self, source_id: str, reference_time: datetime, hours: float = 48.0) -> None:
        try:
            all_buckets = await self.list_all(include_archive=False)
        except Exception:
            return

        rippled = 0
        max_ripple = 5
        for bucket in all_buckets:
            if rippled >= max_ripple:
                break
            if bucket["id"] == source_id:
                continue
            meta = bucket.get("metadata", {})
            if meta.get("pinned") or meta.get("protected") or meta.get("type") in ("permanent", "feel"):
                continue

            created_str = meta.get("created", meta.get("last_active", ""))
            try:
                created = datetime.fromisoformat(str(created_str))
                delta_hours = abs((reference_time - created).total_seconds()) / 3600
            except (ValueError, TypeError):
                continue

            if delta_hours <= hours:
                file_path = self._find_bucket_file(bucket["id"])
                if not file_path:
                    continue
                try:
                    post = self._safe_load_post(file_path)
                    current_count = post.get("activation_count", 1)
                    post["activation_count"] = round(current_count + 0.3, 1)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(frontmatter.dumps(post))
                    rippled += 1
                except Exception:
                    continue

    # ---------------------------------------------------------
    # Multi-dimensional search (core feature)
    # ---------------------------------------------------------
    async def search(
        self,
        query: str,
        limit: int = None,
        domain_filter: list[str] = None,
        query_valence: float = None,
        query_arousal: float = None,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        limit = limit or self.max_results
        all_buckets = await self.list_all(include_archive=False)

        if not all_buckets:
            return []

        # --- 修复域过滤的脆弱迭代 ---
        if domain_filter:
            filter_set = {str(d).lower() for d in domain_filter}
            candidates = []
            for b in all_buckets:
                b_domain = b["metadata"].get("domain", [])
                if isinstance(b_domain, str):
                    b_domain = [b_domain]
                elif not isinstance(b_domain, list):
                    b_domain = []
                if {str(d).lower() for d in b_domain} & filter_set:
                    candidates.append(b)
            if not candidates:
                # domain_filter 没有匹配到任何桶时，严格返回空，
                # 而不是退化成搜全部（避免用户以为过滤生效但实际没过滤）
                logger.info(
                    f"domain_filter {domain_filter} matched no buckets, returning empty"
                )
                return []
        else:
            candidates = all_buckets

        scored = []
        for bucket in candidates:
            meta = bucket.get("metadata", {})

            try:
                topic_score = self._calc_topic_score(query, bucket)
                emotion_score = self._calc_emotion_score(query_valence, query_arousal, meta)
                time_score = self._calc_time_score(meta)
                importance_score = max(1, min(10, int(meta.get("importance", 5)))) / 10.0

                total = (
                    topic_score * self.w_topic
                    + emotion_score * self.w_emotion
                    + time_score * self.w_time
                    + importance_score * self.w_importance
                )
                
                weight_sum = self.w_topic + self.w_emotion + self.w_time + self.w_importance
                normalized = (total / weight_sum) * 100 if weight_sum > 0 else 0

                if meta.get("resolved", False):
                    normalized *= 0.3

                if normalized >= self.fuzzy_threshold:
                    bucket["score"] = round(normalized, 2)
                    scored.append(bucket)
            except Exception as e:
                logger.warning(
                    f"Scoring failed for bucket {bucket.get('id', '?')} / "
                    f"桶评分失败: {e}"
                )
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ---------------------------------------------------------
    # Topic relevance sub-score (REWRITTEN: MAX-WIN + JIEBA SEGMENTATION + TYPE SAFETY)
    # 彻底重写的文本相关性算法：双向切词短路 + 最高分原则 + 极致类型安全
    # ---------------------------------------------------------
    def _calc_topic_score(self, query: str, bucket: dict) -> float:
        """
        Calculate text dimension relevance score (0~1).
        计算文本维度的相关性得分（强化版安全机制 + Jieba 切词）。
        """
        meta = bucket.get("metadata", {})
        query_lower = query.lower()

        name = meta.get("name", "")
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        domain = meta.get("domain", [])
        if isinstance(domain, str):
            domain = [domain]
            
        content = str(bucket.get("content", ""))[:1000]

        name_lower = str(name).lower()
        tags_lower = [str(t).lower() for t in tags]

        # --- 0. 最强绝对命中机制（全句包含） ---
        if query_lower in name_lower or name_lower in query_lower:
            return 1.0
        for t in tags_lower:
            if query_lower in t or t in query_lower:
                return 1.0

        # --- 1. 核心词短路机制（Jieba 分词检测） ---
        try:
            words = list(jieba.cut(query_lower))
        except Exception:
            words = query_lower.split()

        query_parts = [
            p.strip() for p in words 
            if p.strip() and p.strip() not in self.wikilink_stopwords
        ]
        if not query_parts:
            query_parts = [query_lower]

        for part in query_parts:
            if name_lower and part in name_lower:
                return 1.0
            for t in tags_lower:
                if t and part in t:
                    return 1.0

        # --- 2. 模糊匹配得分独立计算（强制转换为字符串，杜绝 Exception 静默拦截） ---
        name_score = fuzz.partial_ratio(query, str(name)) / 100.0 if name else 0.0
        domain_score = max([fuzz.partial_ratio(query, str(d)) for d in domain] + [0]) / 100.0 if domain else 0.0
        tag_score = max([fuzz.partial_ratio(query, str(t)) for t in tags] + [0]) / 100.0 if tags else 0.0
        content_score = fuzz.partial_ratio(query, content) / 100.0 if content else 0.0

        # --- 3. 最高亮机制（Max-Win）替代加权平均 ---
        final_score = max(
            name_score,
            tag_score,
            domain_score * 0.9,     # 域的匹配权重略微打折
            content_score * 0.8     # 正文冗长，单纯匹配的权重垫底
        )

        return final_score

    # ---------------------------------------------------------
    # Emotion resonance sub-score
    # ---------------------------------------------------------
    def _calc_emotion_score(
        self, q_valence: float, q_arousal: float, meta: dict
    ) -> float:
        if q_valence is None or q_arousal is None:
            return 0.5

        try:
            b_valence = float(meta.get("valence", 0.5))
            b_arousal = float(meta.get("arousal", 0.3))
        except (ValueError, TypeError):
            return 0.5

        dist = math.sqrt((q_valence - b_valence) ** 2 + (q_arousal - b_arousal) ** 2)
        return max(0.0, 1.0 - dist / 1.414)

    # ---------------------------------------------------------
    # Time proximity sub-score
    # ---------------------------------------------------------
    def _calc_time_score(self, meta: dict) -> float:
        last_active_str = meta.get("last_active", meta.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days = 30
        return math.exp(-0.1 * days)

    # ---------------------------------------------------------
    # List all buckets
    # ---------------------------------------------------------
    async def list_all(self, include_archive: bool = False) -> list[dict]:
        buckets = []
        dirs = [self.permanent_dir, self.dynamic_dir, self.feel_dir]
        if include_archive:
            dirs.append(self.archive_dir)

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    file_path = os.path.join(root, filename)
                    bucket = self._load_bucket(file_path)
                    if bucket:
                        buckets.append(bucket)

        return buckets

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    async def get_stats(self) -> dict:
        stats = {
            "permanent_count": 0,
            "dynamic_count": 0,
            "archive_count": 0,
            "feel_count": 0,
            "total_size_kb": 0.0,
            "domains": {},
        }

        for subdir, key in [
            (self.permanent_dir, "permanent_count"),
            (self.dynamic_dir, "dynamic_count"),
            (self.archive_dir, "archive_count"),
            (self.feel_dir, "feel_count"),
        ]:
            if not os.path.exists(subdir):
                continue
            for root, _, files in os.walk(subdir):
                for f in files:
                    if f.endswith(".md"):
                        stats[key] += 1
                        fpath = os.path.join(root, f)
                        try:
                            stats["total_size_kb"] += os.path.getsize(fpath) / 1024
                        except OSError:
                            pass
                        domain_name = os.path.basename(root)
                        if domain_name != os.path.basename(subdir):
                            stats["domains"][domain_name] = stats["domains"].get(domain_name, 0) + 1

        return stats

    # ---------------------------------------------------------
    # Archive bucket
    # ---------------------------------------------------------
    async def archive(self, bucket_id: str) -> bool:
        file_path = self._find_bucket_file(bucket_id)
        if not file_path:
            return False

        try:
            post = self._safe_load_post(file_path)
            domain = post.get("domain", ["未分类"])
            primary_domain = sanitize_name(domain[0]) if domain else "未分类"
            archive_subdir = os.path.join(self.archive_dir, primary_domain)
            os.makedirs(archive_subdir, exist_ok=True)

            dest = safe_path(archive_subdir, os.path.basename(file_path))

            post["type"] = "archived"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

            shutil.move(file_path, str(dest))
        except Exception as e:
            logger.error(
                f"Failed to archive bucket / 归档桶失败: {bucket_id}: {e}"
            )
            return False

        logger.info(f"Archived bucket / 归档记忆桶: {bucket_id} → archive/{primary_domain}/")
        return True

    # ---------------------------------------------------------
    # Internal: find bucket file
    # ---------------------------------------------------------
    def _find_bucket_file(self, bucket_id: str) -> Optional[str]:
        if not bucket_id:
            return None
        for dir_path in [self.permanent_dir, self.dynamic_dir, self.archive_dir, self.feel_dir]:
            if not os.path.exists(dir_path):
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    name_part = fname[:-3]
                    if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                        return os.path.join(root, fname)
        return None

    # ---------------------------------------------------------
    # Internal: load bucket data
    # ---------------------------------------------------------
    def _safe_load_post(self, file_path: str):
        """
        Wrap frontmatter.load to tolerate dirty YAML headers that contain
        a 'content' key (would collide with the body positional arg).
        Strategy: try native load first; on collision, manually strip the
        offending key from YAML and rebuild the Post object.
        包装 frontmatter.load 以容忍 YAML 头里混入 'content' 键的脏数据
        （会和 body 位置参数撞键）。策略：先尝试原生 load；如果撞键，
        手动从 YAML 里剥掉冲突字段后重建 Post 对象。
        """
        try:
            return frontmatter.load(file_path)
        except TypeError as e:
            if "content" not in str(e):
                raise
            # Manual repair: split YAML header, drop 'content' key, rebuild
            # 手动修复：拆开 YAML 头，丢掉 content 键，重组
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.startswith("---\n"):
                raise
            end = text.find("\n---\n", 4)
            if end < 0:
                raise
            yaml_part = text[4:end]
            body = text[end + 5:]
            # Remove 'content:' line and any continuation lines until next top-level key
            # 删 content: 行及其续行，直到下一个顶级 YAML 字段
            cleaned_lines = []
            skip = False
            for line in yaml_part.splitlines(keepends=True):
                if skip:
                    # continuation if line starts with whitespace; otherwise stop skipping
                    if line and line[0] in " \t":
                        continue
                    skip = False
                if line.startswith("content:"):
                    skip = True
                    continue
                cleaned_lines.append(line)
            cleaned_yaml = "".join(cleaned_lines)
            cleaned_text = "---\n" + cleaned_yaml + "---\n" + body
            logger.warning(
                f"Auto-cleaned 'content' from YAML header / "
                f"自动清理YAML头里的content键: {file_path}"
            )
            return frontmatter.loads(cleaned_text)

    def _load_bucket(self, file_path: str) -> Optional[dict]:
        try:
            post = self._safe_load_post(file_path)
            return {
                "id": post.get("id", Path(file_path).stem),
                "metadata": dict(post.metadata),
                "content": post.content,
                "path": file_path,
            }
        except Exception as e:
            logger.warning(
                f"Failed to load bucket file / 加载桶文件失败: {file_path}: {e}"
            )
            return None 
