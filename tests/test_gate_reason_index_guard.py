"""复核用：候选数多于分数时 record_rejections 会不会炸。

server.py:3348 的 `scores[index] if index < len(scores) else 0` 说明这条路上
buckets 可能比 scores 长；record_rejections 用的是裸索引。
"""
import gate_reason_receipt as g


def test_record_rejections_survives_more_buckets_than_scores():
    state, token = g.begin()
    try:
        buckets = [
            {"id": "a", "_ds_gate_final": 10},
            {"id": "b", "_ds_gate_final": 10},
            {"id": "c", "_ds_gate_final": 10},
        ]
        scores = [50, 50]
        g.record_rejections(buckets, scores, threshold=60)
        receipt = state.get("receipt")
        assert receipt is not None
        assert len(receipt["items"]) + receipt["omitted"] == 3
    finally:
        g.reset(token)


def test_record_rejections_survives_short_reasons_vector():
    state, token = g.begin()
    try:
        buckets = [
            {"id": "a", "_ds_gate_final": 10},
            {"id": "b", "_ds_gate_final": 10},
        ]
        scores = g.Scores([50, 50], [g.missing("absent_in_response")])
        g.record_rejections(buckets, scores, threshold=60)
        assert state.get("receipt") is not None
    finally:
        g.reset(token)
