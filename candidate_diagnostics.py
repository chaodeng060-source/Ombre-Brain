"""Request-local, ID-only observations of the real breath candidate pipeline.

No storage or provider clients live here. A snapshot is evidence of one stage,
not a claim that a target absent from a top-k is absent from an index.
"""
from contextvars import ContextVar
import math
import os


current = ContextVar("breath_candidate_diagnostics", default=None)


def enabled():
    return os.environ.get("OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


class CandidateDiagnostics:
    def __init__(self, target_ids):
        self.target_ids = tuple(dict.fromkeys(target_ids))
        self.stages = []
        self.rejections = {bid: [] for bid in self.target_ids}
        self.drops = {bid: [] for bid in self.target_ids}
        self._previous = {}
        self.bm25 = {}
        self.vector = {}
        self.differences = [
            "background_setup_not_started", "query_expansion_skipped",
            "vector_shadow_skipped", "fusion_shadow_skipped",
            "resident_snapshot_only_no_refresh", "retrieval_hints_refresh_skipped",
            "live_chord_not_evaluated", "sqlite_fallback_reads_store_not_live_cache",
        ]

    def observe(self, stage, rows, *, status="ok"):
        positions = {}
        count = 0
        for count, row in enumerate(rows, 1):
            if isinstance(row, dict):
                bid, score = str(row.get("id", "")), row.get("score")
            else:
                bid, score = str(row[0]), row[1]
            if bid in self.rejections:
                positions.setdefault(bid, {"rank": count, "score": number(score)})
        if stage.startswith(("keyword_", "curated_lexical_", "entity_", "state_link_")):
            flow = stage.split("_", 1)[0]
        elif stage in {"vector_raw_top_k", "vector_similarity_floor"}:
            flow = "vector"
        elif stage == "rg_literal":
            flow = "rg_literal"
        else:
            flow = "main"
        previous = self._previous.get(flow, {})
        for bid in previous.keys() - positions.keys():
            self.drops[bid].append({"stage": stage, "previous_rank": previous[bid]["rank"]})
        self._previous[flow] = positions
        self.stages.append({
            "stage": stage, "status": status, "count": count,
            "targets": {bid: positions.get(bid) for bid in self.target_ids},
        })

    def reject(self, bucket, reason):
        bid = str(bucket.get("id", ""))
        if bid in self.rejections and reason not in self.rejections[bid]:
            self.rejections[bid].append(reason)

    def finish(self, matches, state_links):
        partial = any(s["status"] in {"error", "timeout"} for s in self.stages)
        partial = partial or self.bm25.get("status") == "error"
        return {
            "mode": "candidates_only", "completed": not partial, "partial": partial,
            "stopped_before": "gate_and_assembly",
            "candidate_set_may_differ": True, "differences": self.differences,
            "stages": self.stages, "target_rejections": self.rejections,
            "target_drop_observations": self.drops,
            "bm25": self.bm25, "vector": self.vector,
            "candidates": [{"id": str(b["id"]), "score": number(b.get("score"))}
                           for b in matches],
            "state_links": [{"id": str(b["id"]), "score": number(b.get("score"))}
                            for b in state_links],
            "relation": {"status": "not_executed_post_gate", "targets": {
                bid: None for bid in self.target_ids}},
        }


def observe(stage, rows, *, status="ok"):
    diagnostic = current.get()
    if diagnostic is not None:
        diagnostic.observe(stage, rows, status=status)


def reject(bucket, reason):
    diagnostic = current.get()
    if diagnostic is not None:
        diagnostic.reject(bucket, reason)
