"""E 轴结果回填（e_outcome）的契约。

立场：outcome 是事后回填的「这个姿态管不管用」，它只被**透出**给主 agent 看，
不参与排序、不改投票结果。让 backfired 自动降权就是拿统计发明排名——正是
上游 test_e_axis_ownership.py 钉死要禁的那件事（数值情绪不发明初始排名）。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import E_OUTCOMES  # noqa: E402
from utils import EOutcomeGuardError  # noqa: E402
from e_axis_recall import (  # noqa: E402
    ActiveEAnnotation,
    derive_response_posture,
    format_response_posture,
)


def _annotation(
    bucket_id: str,
    *,
    tendency: str = "comfort",
    outcome: str = "",
    priority: int = 70,
) -> ActiveEAnnotation:
    return ActiveEAnnotation(
        bucket_id=bucket_id,
        source_digest="digest-" + bucket_id,
        valence=-0.4,
        arousal=0.5,
        tension=0.5,
        confidence=1.0,
        response_tendency=tendency,
        growth_delta="stable",
        rubric_version="primary-authored/v1",
        scored_at="2026-09-01T10:00:00+08:00",
        authored_by="哥哥",
        initial_priority=priority,
        outcome=outcome,
    )


def test_outcome_vocabulary_is_closed():
    """值域是封闭的三选一，别的词一律不认。"""
    assert E_OUTCOMES == frozenset({"worked", "backfired", "unknown"})


def test_backfired_does_not_change_which_posture_wins():
    """核心契约：把 outcome 从空改成 backfired，胜出姿态必须一模一样。

    这条一旦红，说明有人让 outcome 参与了排序/投票——那是拿统计发明排名。
    """
    clean = [
        (_annotation("a", tendency="comfort"), 0.9),
        (_annotation("b", tendency="comfort"), 0.8),
        (_annotation("c", tendency="engage"), 0.85),
    ]
    marked = [
        (_annotation("a", tendency="comfort", outcome="backfired"), 0.9),
        (_annotation("b", tendency="comfort", outcome="backfired"), 0.8),
        (_annotation("c", tendency="engage", outcome="worked"), 0.85),
    ]
    before = derive_response_posture(clean)
    after = derive_response_posture(marked)
    assert before is not None and after is not None
    assert before.tendency == after.tendency
    assert before.growth_delta == after.growth_delta
    assert before.confidence == after.confidence
    assert before.evidence_count == after.evidence_count
    # 唯一该变的就是这个计数
    assert before.backfired_count == 0
    assert after.backfired_count == 2


def test_backfired_count_only_counts_the_winning_tendency():
    """只数投给胜出姿态的那些——混进别的姿态会把警示稀释成噪音。"""
    posture = derive_response_posture([
        (_annotation("a", tendency="comfort"), 0.9),
        (_annotation("b", tendency="comfort"), 0.9),
        (_annotation("c", tendency="withdraw", outcome="backfired"), 0.2),
    ])
    assert posture is not None
    assert posture.tendency == "comfort"
    # 那条 backfired 投的是 withdraw，不该算进 comfort 的警示
    assert posture.backfired_count == 0


def test_posture_block_surfaces_the_warning():
    """砸过就要在姿态块里说出来，不许沉默。"""
    posture = derive_response_posture([
        (_annotation("a", tendency="comfort", outcome="backfired"), 0.9),
    ])
    assert posture is not None
    rendered = format_response_posture(posture, activation_id="act-1")
    assert "[backfired:1]" in rendered
    assert "backfired" in rendered
    assert "上次是怎么砸的" in rendered


def test_clean_posture_block_has_no_warning_noise():
    """没砸过就别加戏——通知刷屏会把真提醒盖成背景音。"""
    posture = derive_response_posture([
        (_annotation("a", tendency="comfort", outcome="worked"), 0.9),
    ])
    assert posture is not None
    rendered = format_response_posture(posture, activation_id="act-1")
    assert "[backfired:0]" in rendered
    assert "⚠" not in rendered


@pytest.mark.parametrize("bad", ["", "  ", "worked ", "WORKED", "good", "失败"])
def test_unknown_outcome_never_counts_as_backfired(bad: str):
    """只有精确的 backfired 才触发警示，脏值不许蒙混过关。"""
    posture = derive_response_posture([
        (_annotation("a", tendency="comfort", outcome=bad), 0.9),
    ])
    assert posture is not None
    assert posture.backfired_count == 0


# ---------------------------------------------------------------
# 写入侧：回填只能落定一次
# ---------------------------------------------------------------


async def _make_e_bucket(bucket_mgr) -> str:
    return await bucket_mgr.create(
        content="测试用 E：她说累的时候先接住人。",
        name="outcome-seal-probe",
        domain=["关系"],
        e_authored_by="哥哥",
        e_initial_priority=70,
        e_valence=-0.3,
        e_arousal=0.5,
        e_tension=0.5,
        e_confidence=1.0,
        e_response_tendency="comfort",
        e_growth_delta="stable",
    )


@pytest.mark.asyncio
async def test_outcome_can_be_written_once(bucket_mgr):
    bucket_id = await _make_e_bucket(bucket_mgr)
    assert await bucket_mgr.update(bucket_id, e_outcome="worked") is True


@pytest.mark.asyncio
async def test_outcome_is_sealed_against_reversal(bucket_mgr):
    """改判必须走后继 E，不许事后改分——这是整个回填机制可信的前提。"""
    bucket_id = await _make_e_bucket(bucket_mgr)
    await bucket_mgr.update(bucket_id, e_outcome="worked")
    with pytest.raises(EOutcomeGuardError, match="sealed"):
        await bucket_mgr.update(bucket_id, e_outcome="backfired")


@pytest.mark.asyncio
async def test_rewriting_the_same_outcome_is_idempotent(bucket_mgr):
    """重试/重放同一个判定不该炸——封的是改判，不是重复写。"""
    bucket_id = await _make_e_bucket(bucket_mgr)
    await bucket_mgr.update(bucket_id, e_outcome="worked")
    assert await bucket_mgr.update(bucket_id, e_outcome="worked") is True


@pytest.mark.asyncio
async def test_outcome_rejects_values_outside_the_vocabulary(bucket_mgr):
    bucket_id = await _make_e_bucket(bucket_mgr)
    with pytest.raises(EOutcomeGuardError, match="invalid e_outcome"):
        await bucket_mgr.update(bucket_id, e_outcome="很好")


@pytest.mark.asyncio
async def test_outcome_only_applies_to_primary_authored_e(bucket_mgr):
    """普通桶没有「姿态」可言，给它回填结果是概念错位。"""
    bucket_id = await bucket_mgr.create(
        content="普通记忆桶，不是 E。",
        name="plain-bucket",
        domain=["工程"],
    )
    with pytest.raises(EOutcomeGuardError, match="primary-authored E"):
        await bucket_mgr.update(bucket_id, e_outcome="worked")
