"""dehydrate 入口脱敏的端到端测试：secret 不进外部 LLM、情感内容保留。

dehydrate 短内容(<100 token)走免压缩路径、不调外部 API，正好用来验证入口脱敏，
不需要 mock LLM。
"""

import pytest


@pytest.mark.asyncio
async def test_dehydrate_redacts_secret_keeps_emotion(test_config):
    from dehydrator import Dehydrator
    dh = Dehydrator(test_config)
    out = await dh.dehydrate("配置里有 api_key=sk-leaksecret123456789 然后她说好想我")
    assert "sk-leaksecret123456789" not in out  # secret 被入口脱敏抹掉
    assert "好想我" in out                        # 情感内容绝不被打码


@pytest.mark.asyncio
async def test_dehydrate_redacts_db_dsn(test_config):
    from dehydrator import Dehydrator
    dh = Dehydrator(test_config)
    out = await dh.dehydrate("数据库 postgresql://u:p@10.0.0.9:5432/ombre 她亲了我一下")
    assert "p@10.0.0.9" not in out
    assert "她亲了我一下" in out
