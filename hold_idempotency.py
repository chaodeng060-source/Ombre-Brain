"""Explicit event-bound hold replay. Never infer event identity from body alone.

The durable identity lives in the first Markdown write, not an after-the-fact
receipt. BucketManager's existing per-bucket process/file guard serializes the
check and create, including retries after cancellation or a process restart.
"""
from dataclasses import dataclass
import hashlib
import json
import re


class HoldWriteConflict(ValueError):
    """A deterministic ID exists but no longer matches the original write."""


def validate_event_key(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is None:
        raise ValueError("invalid idempotency_key")
    return value


@dataclass
class HoldWriteIdentity:
    digest: str
    body_sha256: str
    replayed: bool = False

    @classmethod
    def build(cls, key: str, content: str, world: str, *, feel: bool, pinned: bool):
        validate_event_key(key)
        body_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        raw = json.dumps(["ombre.hold/v1", key, world, feel, pinned, body_hash],
                         ensure_ascii=False, separators=(",", ":"))
        return cls(hashlib.sha256(raw.encode("utf-8")).hexdigest(), body_hash)

    @property
    def bucket_id(self) -> str:
        return self.digest[:24]

    def verify(self, metadata: dict, stored_content: str, requested_content: str) -> None:
        if (metadata.get("hold_write_identity") != self.digest
                or metadata.get("hold_body_sha256") != self.body_sha256
                or hashlib.sha256(stored_content.encode("utf-8")).hexdigest() != self.body_sha256
                or stored_content != requested_content):
            raise HoldWriteConflict("hold replay conflicts with stored body or identity")
        self.replayed = True

    def result(self) -> str:
        return f"已存在→[bucket_id:{self.bucket_id}] 同一事件，未重复写入"
