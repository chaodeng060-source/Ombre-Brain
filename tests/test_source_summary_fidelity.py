import json

from source_summary_fidelity import guarded_summary


def test_summary_cannot_turn_border_collie_into_cat():
    source = '朝灯说小区的咖啡是一只九岁多的黑白边牧，见面会舔她嘴巴。'
    bad = json.dumps({'core_facts': ['九岁猫咖啡舔朝灯嘴巴'], 'summary': '九岁猫'}, ensure_ascii=False)
    repaired = json.loads(guarded_summary(bad, source))
    assert '猫' not in str(repaired)
    assert repaired['core_facts'] == [source]


def test_real_cat_and_dog_are_preserved():
    source = '奶咖是猫，卡兜是狗。'
    summary = json.dumps({'core_facts': ['奶咖是猫，卡兜是狗'], 'summary': '奶咖与卡兜'}, ensure_ascii=False)
    assert guarded_summary(summary, source) == summary
