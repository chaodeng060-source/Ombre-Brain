"""Deterministic entity/alias sidecar for memory buckets.

The entity layer is deliberately separate from Markdown bucket schemas.  It is
fed by explicit configuration seeds and deterministic write-time mentions; an
LLM is never allowed to invent aliases.  Alias collisions are preserved and a
query resolves only when exactly one entity owns the normalized alias.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import unicodedata
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence


ENTITY_TYPES = frozenset({"person", "place", "project"})
_MAX_TERM_CHARS = 256
_BUSY_TIMEOUT_MS = 15_000
_DB_NAME = "entities.sqlite3"

# CJK has no general-purpose word boundary in the standard library.  A finite
# blacklist of compounds is unsafe (``老婆饼`` is only one of infinitely many
# examples), so adjacent Han characters are accepted only when they are narrow
# grammatical context characters.  False negatives are intentional: Phase 2's
# contract is to split/miss rather than merge unrelated concepts.
_CJK_LEFT_CONTEXT = frozenset(
    "我你他她它俺咱和与跟同向对给替为找问看听说想念爱祝请让把被从由在到去来见叫陪抱亲"
)
_CJK_RIGHT_CONTEXT = frozenset(
    "的了呢啊呀吧嘛在是有会要想说问看听做去来见叫陪今昨明刚正还又也都就真很更最能可该将已曾让把被给跟和与完喜记合带"
)


class EntityStoreError(RuntimeError):
    """Base error for entity sidecar failures."""


class UnsafeEntityStore(EntityStoreError):
    """The sidecar path is a symlink, hardlink, or has unsafe permissions."""


class EntityStoreUnavailable(EntityStoreError):
    """The write sidecar was not explicitly initialized."""


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryResolution:
    canonical_query: str
    entity_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    ambiguous_terms: tuple[str, ...] = ()


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("entity term must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise ValueError("entity term cannot be empty")
    if "\x00" in normalized:
        raise ValueError("entity term cannot contain NUL")
    return normalized


def normalize_term(value: str) -> str:
    """Normalize names and aliases without doing linguistic inference."""
    normalized = _normalize_text(value)
    if len(normalized) > _MAX_TERM_CHARS:
        raise ValueError("entity term is too long")
    return normalized


def content_sha256(content: str) -> str:
    if not isinstance(content, str):
        raise TypeError("bucket content must be a string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entity_id(entity_type: str, canonical_norm: str) -> str:
    digest = hashlib.sha256(
        f"ombre-entity-v1\0{entity_type}\0{canonical_norm}".encode("utf-8")
    ).hexdigest()
    return f"ent_{digest[:24]}"


def _is_word_char(char: str) -> bool:
    return bool(char) and (char.isalnum() or char == "_")


def _contains_cjk(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def _has_alias_boundary(text: str, start: int, end: int, alias_norm: str) -> bool:
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""

    # Non-CJK aliases use Unicode word boundaries within non-CJK scripts.  A
    # following Chinese verb is a valid boundary ("Vae今天"), while Vae2,
    # myVae, and an accented Latin alias inside a longer Latin word are not.
    if not _contains_cjk(alias_norm):
        if (
            _is_word_char(alias_norm[0])
            and _is_word_char(before)
            and not _contains_cjk(before)
        ):
            return False
        if (
            _is_word_char(alias_norm[-1])
            and _is_word_char(after)
            and not _contains_cjk(after)
        ):
            return False
        return True

    if _is_word_char(before) and not _contains_cjk(before):
        return False
    if _is_word_char(after) and not _contains_cjk(after):
        return False
    if len(alias_norm) == 1 and (
        (before and _contains_cjk(before)) or (after and _contains_cjk(after))
    ):
        return False
    if before and _contains_cjk(before) and before not in _CJK_LEFT_CONTEXT:
        return False
    if after and _contains_cjk(after) and after not in _CJK_RIGHT_CONTEXT:
        return False
    return True


def _find_spans(text: str, alias_norm: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = text.find(alias_norm, offset)
        if start < 0:
            break
        end = start + len(alias_norm)
        if _has_alias_boundary(text, start, end, alias_norm):
            spans.append((start, end))
        offset = start + 1
    return spans


def entity_mention_present(content: str, mention: str) -> bool:
    """Return whether ``mention`` is a boundary-safe exact span in content.

    The write orchestrator uses this same predicate before handing model
    candidates to the store.  Keeping one boundary implementation prevents a
    loose substring prefilter from aborting an otherwise valid seed/hash
    refresh (for example ``老婆`` inside ``老婆饼``).
    """
    try:
        return bool(_find_spans(_normalize_text(content), normalize_term(mention)))
    except (TypeError, ValueError):
        return False


class EntityStore:
    """SQLite sidecar with conservative entity resolution.

    ``initialize=False`` is useful in read-only consumers.  Query methods never
    call the initializer and open SQLite with ``mode=ro``; a missing or corrupt
    store simply produces an empty resolution rather than creating files.
    """

    def __init__(self, config: dict, *, initialize: bool = True):
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        buckets_dir = config.get("buckets_dir")
        if not isinstance(buckets_dir, (str, os.PathLike)) or not os.fspath(buckets_dir):
            raise ValueError("config.buckets_dir is required")
        self.buckets_dir = os.path.abspath(os.fspath(buckets_dir))
        self.entities_dir = os.path.join(self.buckets_dir, ".entities")
        self.db_path = os.path.join(self.entities_dir, _DB_NAME)
        self.config = config.get("entities", {}) or {}
        if not isinstance(self.config, dict):
            raise ValueError("config.entities must be a mapping")
        if initialize:
            self.initialize()

    # ------------------------------------------------------------------
    # Secure layout and connections
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_regular_private(path: str, *, exact_mode: int = 0o600) -> None:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise UnsafeEntityStore(f"refusing non-regular entity sidecar: {path}")
        if info.st_nlink != 1:
            raise UnsafeEntityStore(f"refusing hardlinked entity sidecar: {path}")
        if stat.S_IMODE(info.st_mode) != exact_mode:
            raise UnsafeEntityStore(f"unsafe entity sidecar mode: {path}")

    @staticmethod
    def _assert_private_dir(path: str) -> None:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise UnsafeEntityStore(f"refusing non-directory entity sidecar: {path}")
        if stat.S_IMODE(info.st_mode) != 0o700:
            raise UnsafeEntityStore(f"unsafe entity sidecar directory mode: {path}")

    def _ensure_layout_for_initialize(self) -> None:
        os.makedirs(self.buckets_dir, exist_ok=True)
        try:
            os.mkdir(self.entities_dir, 0o700)
        except FileExistsError:
            info = os.lstat(self.entities_dir)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise UnsafeEntityStore("refusing symlink/non-directory .entities")
            os.chmod(self.entities_dir, 0o700)
        self._assert_private_dir(self.entities_dir)

        if not os.path.lexists(self.db_path):
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(self.db_path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        if not os.path.lexists(self.db_path):
            raise EntityStoreUnavailable("entity database could not be created")
        self._assert_no_link_then_chmod(self.db_path)

    def _assert_no_link_then_chmod(self, path: str) -> None:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise UnsafeEntityStore(f"refusing non-regular entity sidecar: {path}")
        if info.st_nlink != 1:
            raise UnsafeEntityStore(f"refusing hardlinked entity sidecar: {path}")
        os.chmod(path, 0o600)
        self._assert_regular_private(path)

    def _validate_read_layout(self) -> None:
        self._assert_private_dir(self.entities_dir)
        self._assert_regular_private(self.db_path)
        for suffix in ("-wal", "-shm"):
            sidecar = self.db_path + suffix
            if os.path.lexists(sidecar):
                self._assert_regular_private(sidecar)

    def _enforce_write_modes(self) -> None:
        os.chmod(self.entities_dir, 0o700)
        self._assert_private_dir(self.entities_dir)
        for path in (self.db_path, self.db_path + "-wal", self.db_path + "-shm"):
            if os.path.lexists(path):
                self._assert_no_link_then_chmod(path)

    def _connect_write(self) -> sqlite3.Connection:
        if not os.path.lexists(self.db_path):
            raise EntityStoreUnavailable("entity store is not initialized")
        self._assert_private_dir(self.entities_dir)
        self._assert_regular_private(self.db_path)
        conn = sqlite3.connect(
            self.db_path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA trusted_schema = OFF")
            # WAL readers may create/update ``-shm`` even when SQLite is
            # opened with mode=ro.  Entity recall is required to be strictly
            # filesystem-read-only, so use the rollback journal instead.
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("PRAGMA synchronous = FULL")
            return conn
        except Exception:
            conn.close()
            raise

    def _connect_read(self) -> sqlite3.Connection:
        self._validate_read_layout()
        uri = f"file:{urllib.parse.quote(self.db_path, safe='/')}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA trusted_schema = OFF")
            conn.execute("PRAGMA query_only = ON")
            return conn
        except Exception:
            conn.close()
            raise

    # ------------------------------------------------------------------
    # Schema and seeds
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        self._ensure_layout_for_initialize()
        conn = self._connect_write()
        try:
            conn.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    canonical_norm TEXT NOT NULL,
                    entity_type TEXT NOT NULL CHECK (
                        entity_type IN ('person', 'place', 'project')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (entity_type, canonical_norm)
                );

                CREATE TABLE IF NOT EXISTS entity_aliases (
                    entity_id TEXT NOT NULL REFERENCES entities(entity_id)
                        ON DELETE CASCADE,
                    alias TEXT NOT NULL,
                    alias_norm TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (entity_id, alias_norm)
                );
                CREATE INDEX IF NOT EXISTS idx_entity_aliases_norm
                    ON entity_aliases(alias_norm);

                CREATE TABLE IF NOT EXISTS bucket_entities (
                    bucket_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL REFERENCES entities(entity_id)
                        ON DELETE CASCADE,
                    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
                    linked_at TEXT NOT NULL,
                    PRIMARY KEY (bucket_id, entity_id)
                );
                CREATE INDEX IF NOT EXISTS idx_bucket_entities_entity
                    ON bucket_entities(entity_id, bucket_id);

                CREATE TABLE IF NOT EXISTS entity_events (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entity_id TEXT REFERENCES entities(entity_id)
                        ON DELETE SET NULL,
                    bucket_id TEXT,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_events_entity_time
                    ON entity_events(entity_id, occurred_at);
                CREATE INDEX IF NOT EXISTS idx_entity_events_bucket_time
                    ON entity_events(bucket_id, occurred_at);
                COMMIT;
                """
            )
            conn.execute("BEGIN IMMEDIATE")
            for seed in self._iter_seeds(self.config.get("seeds", ())):
                self._resolve_or_create_in_conn(
                    conn,
                    seed["canonical_name"],
                    aliases=seed["aliases"],
                    entity_type=seed["entity_type"],
                    event_action="seed_entity",
                )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
            self._enforce_write_modes()

    @staticmethod
    def _iter_seeds(raw) -> list[dict]:
        if raw in (None, (), [], {}):
            return []
        seeds: list[dict] = []
        if isinstance(raw, Mapping):
            items = []
            for canonical_name, spec in raw.items():
                if isinstance(spec, Mapping):
                    value = dict(spec)
                    value.setdefault("canonical_name", canonical_name)
                else:
                    value = {"canonical_name": canonical_name, "aliases": spec}
                items.append(value)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            items = list(raw)
        else:
            raise ValueError("entities.seeds must be a list or mapping")

        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("each entity seed must be a mapping")
            canonical_name = item.get("canonical_name", item.get("canonical"))
            entity_type = item.get("type", item.get("entity_type"))
            aliases = item.get("aliases", ())
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, Sequence):
                raise ValueError("seed aliases must be a list")
            seeds.append({
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "aliases": tuple(aliases),
            })
        return seeds

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_type(entity_type: str) -> str:
        value = str(entity_type or "").strip().casefold()
        if value not in ENTITY_TYPES:
            raise ValueError(f"entity type must be one of {sorted(ENTITY_TYPES)}")
        return value

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        action: str,
        *,
        entity_id: str | None = None,
        bucket_id: str | None = None,
        details: Mapping | None = None,
    ) -> None:
        conn.execute(
            """INSERT INTO entity_events
               (event_id, occurred_at, action, entity_id, bucket_id, details_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                _now(),
                action,
                entity_id,
                bucket_id,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            ),
        )

    def _resolve_or_create_in_conn(
        self,
        conn: sqlite3.Connection,
        canonical_name: str,
        *,
        aliases: Iterable[str] = (),
        entity_type: str,
        event_action: str = "create_entity",
    ) -> EntityRecord:
        canonical_norm = normalize_term(canonical_name)
        clean_canonical = unicodedata.normalize("NFKC", canonical_name).strip()
        entity_type = self._validate_type(entity_type)
        row = conn.execute(
            """SELECT entity_id, canonical_name, entity_type
                 FROM entities
                WHERE entity_type = ? AND canonical_norm = ?""",
            (entity_type, canonical_norm),
        ).fetchone()
        created = row is None
        if created:
            entity_id = _entity_id(entity_type, canonical_norm)
            timestamp = _now()
            conn.execute(
                """INSERT INTO entities
                   (entity_id, canonical_name, canonical_norm, entity_type,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entity_id,
                    clean_canonical,
                    canonical_norm,
                    entity_type,
                    timestamp,
                    timestamp,
                ),
            )
            self._event(
                conn,
                event_action,
                entity_id=entity_id,
                details={"canonical_name": clean_canonical, "type": entity_type},
            )
        else:
            entity_id = row["entity_id"]
            clean_canonical = row["canonical_name"]

        all_aliases: list[str] = [clean_canonical]
        for alias in aliases:
            if not isinstance(alias, str):
                raise ValueError("aliases must contain strings")
            all_aliases.append(unicodedata.normalize("NFKC", alias).strip())
        seen: set[str] = set()
        for alias in all_aliases:
            alias_norm = normalize_term(alias)
            if alias_norm in seen:
                continue
            seen.add(alias_norm)
            cursor = conn.execute(
                """INSERT OR IGNORE INTO entity_aliases
                   (entity_id, alias, alias_norm, created_at)
                   VALUES (?, ?, ?, ?)""",
                (entity_id, alias, alias_norm, _now()),
            )
            if cursor.rowcount:
                self._event(
                    conn,
                    "add_alias",
                    entity_id=entity_id,
                    details={"alias": alias},
                )
        alias_rows = conn.execute(
            """SELECT alias FROM entity_aliases
                WHERE entity_id = ? ORDER BY alias_norm""",
            (entity_id,),
        ).fetchall()
        return EntityRecord(
            entity_id=entity_id,
            canonical_name=clean_canonical,
            entity_type=entity_type,
            aliases=tuple(row["alias"] for row in alias_rows),
        )

    def resolve_or_create(
        self,
        canonical_name: str,
        aliases: Iterable[str] = (),
        entity_type: str = "person",
    ) -> EntityRecord:
        conn = self._connect_write()
        try:
            conn.execute("BEGIN IMMEDIATE")
            record = self._resolve_or_create_in_conn(
                conn,
                canonical_name,
                aliases=aliases,
                entity_type=entity_type,
            )
            conn.execute("COMMIT")
            return record
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
            self._enforce_write_modes()

    @staticmethod
    def _alias_catalog(conn: sqlite3.Connection) -> dict[str, list[sqlite3.Row]]:
        rows = conn.execute(
            """SELECT a.alias, a.alias_norm, e.entity_id,
                      e.canonical_name, e.entity_type
                 FROM entity_aliases AS a
                 JOIN entities AS e ON e.entity_id = a.entity_id
                ORDER BY length(a.alias_norm) DESC, a.alias_norm, e.entity_id"""
        ).fetchall()
        catalog: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            catalog.setdefault(row["alias_norm"], []).append(row)
        return catalog

    @staticmethod
    def _select_mentions(
        normalized_text: str,
        catalog: Mapping[str, Sequence[sqlite3.Row]],
    ) -> list[tuple[int, int, str, Sequence[sqlite3.Row]]]:
        candidates: list[tuple[int, int, str, Sequence[sqlite3.Row]]] = []
        for alias_norm, rows in catalog.items():
            for start, end in _find_spans(normalized_text, alias_norm):
                candidates.append((start, end, alias_norm, rows))
        # Longest alias wins when configured aliases overlap.  Ties are stable.
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
        selected: list[tuple[int, int, str, Sequence[sqlite3.Row]]] = []
        for item in candidates:
            if any(item[0] < old[1] and old[0] < item[1] for old in selected):
                continue
            selected.append(item)
        return sorted(selected, key=lambda item: item[0])

    @staticmethod
    def _unique_for_type(
        rows: Sequence[sqlite3.Row], entity_type: str
    ) -> sqlite3.Row | None:
        matching = {row["entity_id"]: row for row in rows if row["entity_type"] == entity_type}
        if len(matching) == 1:
            return next(iter(matching.values()))
        return None

    def resolve_and_link(
        self,
        bucket_id: str,
        content: str,
        candidates: Sequence[Mapping] = (),
    ) -> tuple[EntityRecord, ...]:
        if not isinstance(bucket_id, str) or not bucket_id.strip() or "\x00" in bucket_id:
            raise ValueError("bucket_id must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("candidates must be a list of {mention, type}")

        conn = self._connect_write()
        try:
            conn.execute("BEGIN IMMEDIATE")
            explicit_ids: set[str] = set()
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ValueError("entity candidate must be a mapping")
                if "aliases" in candidate:
                    raise ValueError("write candidates cannot define aliases")
                mention = candidate.get("mention")
                entity_type = self._validate_type(candidate.get("type"))
                mention_norm = normalize_term(mention)
                if not entity_mention_present(content, mention):
                    raise ValueError("entity candidate is not present in bucket content")

                rows = conn.execute(
                    """SELECT a.alias, a.alias_norm, e.entity_id,
                              e.canonical_name, e.entity_type
                         FROM entity_aliases AS a
                         JOIN entities AS e ON e.entity_id = a.entity_id
                        WHERE a.alias_norm = ?""",
                    (mention_norm,),
                ).fetchall()
                # Any alias with more than one owner is ambiguous globally.
                # Do not let a model-provided coarse type or an exact
                # canonical-name coincidence choose one owner: the Phase-2
                # rule is "split rather than merge incorrectly".
                owners = {row["entity_id"] for row in rows}
                if len(owners) > 1:
                    continue
                unique = self._unique_for_type(rows, entity_type)
                if unique is not None:
                    explicit_ids.add(unique["entity_id"])
                    continue
                if owners:
                    # A model type disagreement cannot mint a second owner for
                    # an operator-seeded alias and poison future query
                    # resolution.  Skip it rather than retyping the entity.
                    continue

                # With no global collision, an exact canonical in the
                # requested type can be reused.  Otherwise create a separate
                # typed entity instead of retyping an existing owner.
                exact = conn.execute(
                    """SELECT entity_id FROM entities
                        WHERE canonical_norm = ? AND entity_type = ?""",
                    (mention_norm, entity_type),
                ).fetchone()
                if exact is not None:
                    explicit_ids.add(exact["entity_id"])
                else:
                    record = self._resolve_or_create_in_conn(
                        conn,
                        mention,
                        aliases=(),
                        entity_type=entity_type,
                    )
                    explicit_ids.add(record.entity_id)

            normalized_content = _normalize_text(content)
            catalog = self._alias_catalog(conn)
            resolved_ids = set(explicit_ids)
            for _start, _end, _alias_norm, rows in self._select_mentions(
                normalized_content, catalog
            ):
                unique = {row["entity_id"]: row for row in rows}
                if len(unique) == 1:
                    resolved_ids.add(next(iter(unique)))

            digest = content_sha256(content)
            previous = {
                row["entity_id"]: row["content_sha256"]
                for row in conn.execute(
                    "SELECT entity_id, content_sha256 FROM bucket_entities WHERE bucket_id = ?",
                    (bucket_id,),
                ).fetchall()
            }
            conn.execute("DELETE FROM bucket_entities WHERE bucket_id = ?", (bucket_id,))
            for entity_id in sorted(resolved_ids):
                conn.execute(
                    """INSERT INTO bucket_entities
                       (bucket_id, entity_id, content_sha256, linked_at)
                       VALUES (?, ?, ?, ?)""",
                    (bucket_id, entity_id, digest, _now()),
                )
                if previous.get(entity_id) != digest:
                    self._event(
                        conn,
                        "link_bucket",
                        entity_id=entity_id,
                        bucket_id=bucket_id,
                        details={"content_sha256": digest},
                    )
            for entity_id in sorted(set(previous) - resolved_ids):
                self._event(
                    conn,
                    "unlink_bucket",
                    entity_id=entity_id,
                    bucket_id=bucket_id,
                    details={"reason": "content_changed"},
                )

            rows = []
            if resolved_ids:
                placeholders = ",".join("?" for _ in resolved_ids)
                rows = conn.execute(
                    f"""SELECT entity_id, canonical_name, entity_type
                          FROM entities WHERE entity_id IN ({placeholders})
                          ORDER BY canonical_name, entity_id""",
                    tuple(sorted(resolved_ids)),
                ).fetchall()
            records = tuple(
                EntityRecord(row["entity_id"], row["canonical_name"], row["entity_type"])
                for row in rows
            )
            conn.execute("COMMIT")
            return records
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
            self._enforce_write_modes()

    # ------------------------------------------------------------------
    # Strictly read-only query APIs
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return ""
        return _normalize_text(query)

    def resolve_query(self, query: str) -> QueryResolution:
        normalized_query = self._normalized_query(query)
        if not normalized_query:
            return QueryResolution(canonical_query="")
        try:
            conn = self._connect_read()
        except (OSError, sqlite3.Error, EntityStoreError):
            return QueryResolution(canonical_query=query)
        try:
            catalog = self._alias_catalog(conn)
            mentions = self._select_mentions(normalized_query, catalog)
            entity_ids: list[str] = []
            terms: list[str] = []
            ambiguous: list[str] = []
            replacements: list[tuple[int, int, str]] = []
            for start, end, alias_norm, rows in mentions:
                unique = {row["entity_id"]: row for row in rows}
                if len(unique) != 1:
                    if alias_norm not in ambiguous:
                        ambiguous.append(alias_norm)
                    continue
                row = next(iter(unique.values()))
                if row["entity_id"] not in entity_ids:
                    entity_ids.append(row["entity_id"])
                    terms.append(row["canonical_name"])
                replacements.append((start, end, row["canonical_name"]))

            # Preserve the legacy query byte-for-byte when no audited alias
            # matched.  Normalization exists only for lookup; it must not
            # silently rewrite unrelated keyword/vector searches.
            canonical = query
            if replacements:
                canonical = normalized_query
                for start, end, replacement in reversed(replacements):
                    canonical = canonical[:start] + replacement + canonical[end:]
            return QueryResolution(
                canonical_query=canonical,
                entity_ids=tuple(entity_ids),
                terms=tuple(terms),
                ambiguous_terms=tuple(ambiguous),
            )
        except sqlite3.Error:
            return QueryResolution(canonical_query=query)
        finally:
            conn.close()

    def canonicalize_query(self, query: str) -> str:
        return self.resolve_query(query).canonical_query

    def linked_bucket_ids(
        self,
        query: str | None = None,
        entity_ids: Iterable[str] | None = None,
        content_hashes: Mapping[str, str] | None = None,
    ) -> list[str]:
        ids: list[str] = []
        if entity_ids is not None:
            if isinstance(entity_ids, str):
                ids = [entity_ids]
            else:
                ids = [str(value) for value in entity_ids]
        elif query is not None:
            ids = list(self.resolve_query(query).entity_ids)
        if not ids:
            return []
        ids = sorted(set(ids))
        try:
            conn = self._connect_read()
        except (OSError, sqlite3.Error, EntityStoreError):
            return []
        try:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"""SELECT DISTINCT bucket_id, content_sha256
                      FROM bucket_entities
                     WHERE entity_id IN ({placeholders})
                     ORDER BY bucket_id""",
                tuple(ids),
            ).fetchall()
            result = []
            for row in rows:
                if content_hashes is not None:
                    expected = content_hashes.get(row["bucket_id"])
                    if expected != row["content_sha256"]:
                        continue
                result.append(row["bucket_id"])
            return result
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def link_is_current(self, bucket_id: str, content: str) -> bool:
        """Return true only when the bucket has links for this exact content."""
        digest = content_sha256(content)
        try:
            conn = self._connect_read()
        except (OSError, sqlite3.Error, EntityStoreError):
            return False
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN content_sha256 = ? THEN 1 ELSE 0 END) AS current
                     FROM bucket_entities WHERE bucket_id = ?""",
                (digest, bucket_id),
            ).fetchone()
            return bool(row and row["total"] and row["total"] == row["current"])
        except sqlite3.Error:
            return False
        finally:
            conn.close()

    def unlink_bucket(self, bucket_id: str) -> int:
        """Remove all sidecar links for a deleted bucket, with audit events."""
        if not isinstance(bucket_id, str) or not bucket_id.strip() or "\x00" in bucket_id:
            raise ValueError("bucket_id must be a non-empty string")
        conn = self._connect_write()
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT entity_id FROM bucket_entities WHERE bucket_id = ?",
                (bucket_id,),
            ).fetchall()
            conn.execute("DELETE FROM bucket_entities WHERE bucket_id = ?", (bucket_id,))
            for row in rows:
                self._event(
                    conn,
                    "unlink_bucket",
                    entity_id=row["entity_id"],
                    bucket_id=bucket_id,
                    details={"reason": "bucket_deleted"},
                )
            conn.execute("COMMIT")
            return len(rows)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
            self._enforce_write_modes()


__all__ = [
    "ENTITY_TYPES",
    "EntityRecord",
    "EntityStore",
    "EntityStoreError",
    "EntityStoreUnavailable",
    "QueryResolution",
    "UnsafeEntityStore",
    "content_sha256",
    "entity_mention_present",
    "normalize_term",
]
