from pathlib import Path


def test_bucket_score_is_labeled_as_activity_score():
    dashboard = Path(__file__).resolve().parents[1] / "dashboard.html"
    html = dashboard.read_text(encoding="utf-8")

    assert '<label>活跃度分</label>' in html
    assert '<label>权重分</label>' not in html
