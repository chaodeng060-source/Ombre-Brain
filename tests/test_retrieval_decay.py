# ============================================================
# Tests: retrieval-time forgetting curve (decay_engine.retrieval_decay_factor)
# 检索期遗忘曲线测试 —— 6/10 偷师 ebbingflow 接进 breath 检索排序
# ============================================================
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from decay_engine import DecayEngine  # noqa: E402


def _engine(**retrieval):
    cfg = {"decay": {"retrieval": retrieval}} if retrieval else {"decay": {}}
    return DecayEngine(cfg, bucket_mgr=None)


def _meta(days_old: float, **kw) -> dict:
    created = (datetime.now() - timedelta(days=days_old)).isoformat()
    return {"created": created, "activation_count": 0, **kw}


def test_fresh_bucket_no_decay():
    f = _engine().retrieval_decay_factor(_meta(0))
    assert f > 0.99


def test_half_life_point():
    # 默认半衰期 14 天：14 天前的桶衰到 ~0.5
    f = _engine().retrieval_decay_factor(_meta(14))
    assert 0.45 < f < 0.55


def test_old_unresolved_hits_floor():
    # 老但没 resolved：吃 0.25 保底，不归零
    f = _engine().retrieval_decay_factor(_meta(365))
    assert f == 0.25


def test_old_resolved_sinks_below_floor():
    # resolved 老桶不吃保底——过期自锁真衰下去
    f = _engine().retrieval_decay_factor(_meta(365, resolved=True))
    assert f < 0.01


def test_rehearsal_stretches_half_life():
    # 常被复述的活得久：同龄桶，激活次数多的衰得慢
    quiet = _engine().retrieval_decay_factor(_meta(14, activation_count=0))
    retold = _engine().retrieval_decay_factor(_meta(14, activation_count=31))
    assert retold > quiet


def test_pinned_protected_permanent_feel_exempt():
    eng = _engine()
    for meta in (
        _meta(365, pinned=True),
        _meta(365, protected=True),
        _meta(365, type="permanent"),
        _meta(365, type="feel"),
    ):
        assert eng.retrieval_decay_factor(meta) == 1.0


def test_missing_timestamp_is_neutral():
    assert _engine().retrieval_decay_factor({"activation_count": 3}) == 1.0


def test_apply_blend_is_gentle():
    # w=0.4：最老非 resolved 桶 ×0.7，新桶 ×1.0
    eng = _engine()
    assert abs(eng.apply_retrieval_decay(100.0, _meta(365)) - 70.0) < 0.5
    assert abs(eng.apply_retrieval_decay(100.0, _meta(0)) - 100.0) < 0.5


def test_weight_zero_disables():
    eng = _engine(weight=0.0)
    assert eng.apply_retrieval_decay(100.0, _meta(365)) == 100.0
