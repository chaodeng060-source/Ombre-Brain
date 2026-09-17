import copy
import json

import pytest

from retrieval_hints import Generation
from retrieval_hints_storage import HintsStore, SourceChanged, VersionConflict
from tests.test_retrieval_hints import BODY, payload


def bucket(body=BODY):
    return {"id": "synthetic-a", "content": body, "path": "/tmp/synthetic-a.md",
            "metadata": {"type": "dynamic", "world": "test"}}


class Peer:
    def __init__(self):
        self.rows = {}
        self.fail = False

    def stage(self, row):
        if self.fail:
            raise OSError("unavailable")
        self.rows[(row["bucket_id"], row["version"])] = {**row, "ready": False}

    def get(self, bid, version):
        return self.rows.get((bid, version))

    def ready(self, bid, version):
        self.rows[(bid, version)]["ready"] = True


def test_registration_and_reservation_are_idempotent(tmp_path):
    store = HintsStore(tmp_path / "hints")
    assert not store.path.exists()
    key = store.register(bucket(), kind="new_write")
    assert store.register(bucket(), kind="backfill") == key
    assert store.reserve(key)
    assert not store.reserve(key)
    assert store.jobs()[0]["kind"] == "new_write"
    # A crash after reservation may have spent the call; it must not retry it.
    assert HintsStore(tmp_path / "hints").reserve(key) is False


def test_pending_is_invisible_and_pg_failure_resumes_without_regeneration(tmp_path):
    store, peer = HintsStore(tmp_path / "hints"), Peer()
    current = bucket()
    version = store.stage(current, Generation("ok", payload(), {"output_tokens": 12}))
    assert store.lookup(current) is None
    peer.fail = True
    with pytest.raises(OSError):
        store.publish(current["id"], version, peer, lambda: current)
    assert store.lookup(current) is None
    peer.fail = False
    store.publish(current["id"], version, peer, lambda: current)
    assert store.lookup(current)["payload"] == payload()
    assert peer.get(current["id"], version)["ready"]
    assert store.pending() == []


def test_reject_source_changed_or_archived(tmp_path):
    store, peer = HintsStore(tmp_path / "hints"), Peer()
    current = bucket()
    version = store.stage(current, Generation("ok", payload()))
    for changed in [None, bucket(BODY + "改变"), {**current, "metadata": {"type": "archived"}}]:
        with pytest.raises(SourceChanged):
            store.publish(current["id"], version, peer, lambda: changed)
        assert store.lookup(current) is None


def test_change_between_pg_and_sqlite_is_not_published(tmp_path):
    store, peer = HintsStore(tmp_path / "hints"), Peer()
    current = bucket()
    version = store.stage(current, Generation("ok", payload()))
    reads = iter([current, bucket(BODY + "changed")])
    with pytest.raises(SourceChanged):
        store.publish(current["id"], version, peer, lambda: next(reads))
    assert peer.get(current["id"], version)["ready"]
    assert store.lookup(current) is None  # PG-ready alone is NOT publication.


def test_old_version_survives_until_both_sides_agree(tmp_path):
    store, peer = HintsStore(tmp_path / "hints"), Peer()
    current = bucket()
    first = store.stage(current, Generation("ok", payload()))
    store.publish(current["id"], first, peer, lambda: current)
    changed_payload = copy.deepcopy(payload())
    changed_payload["aliases"] = []
    second = store.stage(current, Generation("ok", changed_payload))
    peer.fail = True
    with pytest.raises(OSError):
        store.publish(current["id"], second, peer, lambda: current)
    assert store.lookup(current)["version"] == first
    peer.fail = False
    store.publish(current["id"], second, peer, lambda: current)
    assert store.lookup(current)["version"] == second
    store.rollback_pointer(current["id"], expected=second, previous=first)
    assert store.lookup(current)["version"] == first
    with pytest.raises(VersionConflict):
        store.rollback_pointer(current["id"], expected=second, previous=None)


def test_lookup_revalidates_body_and_manifest_does_not_claim_chat_lines(tmp_path):
    store, peer = HintsStore(tmp_path / "hints"), Peer()
    current = bucket()
    version = store.stage(current, Generation("ok", payload()))
    store.publish(current["id"], version, peer, lambda: current)
    assert store.lookup(bucket(BODY + "changed")) is None
    manifest = store.manifest([current, bucket(BODY + "changed")])
    assert len(manifest["records"]) == 1
    row = manifest["records"][0]
    assert row["source_kind"] == "bucket_body"
    assert row["source_link_status"] == "missing"
    assert "speaker" not in row and "jsonl_line" not in row
    assert json.loads(json.dumps(manifest, ensure_ascii=False)) == manifest


def test_corrupted_peer_payload_does_not_publish(tmp_path):
    class CorruptPeer(Peer):
        def get(self, bid, version):
            return {**super().get(bid, version), "payload": {}}
    store, peer = HintsStore(tmp_path / "hints"), CorruptPeer()
    version = store.stage(bucket(), Generation("ok", payload()))
    with pytest.raises(VersionConflict):
        store.publish("synthetic-a", version, peer, bucket)
    assert store.lookup(bucket()) is None
