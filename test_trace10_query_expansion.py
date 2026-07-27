from intent_recall import resolve_intent_recall_policy
from utils import load_config


def _intent(query):
    return resolve_intent_recall_policy(query, {}, 8, 1)["intent"]


def test_trace10_intent_examples():
    assert _intent("什么时候搬到VPS的") == "temporal"
    assert _intent("回忆一下六月底发生的事") == "recall"
    assert _intent("工作台") == "recall"
    assert _intent("想不起来上次吵架的原因了") == "temporal"


def test_legacy_query_expand_alias(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "query_expand:\n"
        "  enabled: true\n"
        "  max_angles: 1\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["query_expansion"]["enabled"] is True
    assert config["query_expansion"]["max_angles"] == 1


def test_query_expansion_preferred_over_legacy_alias(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "query_expand:\n"
        "  enabled: true\n"
        "query_expansion:\n"
        "  enabled: false\n"
        "  max_angles: 3\n",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config["query_expansion"]["enabled"] is False
    assert config["query_expansion"]["max_angles"] == 3
