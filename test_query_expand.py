"""query_expand 自测 —— 假 client，不打真 API。"""
import asyncio
from query_expand import expand_query, _parse_angles


class FakeMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})


class FakeResp:
    def __init__(self, content): self.choices = [FakeMsg(content)]


class FakeClient:
    """可配置返回内容 / 抛异常 / 空 choices 的假 OpenAI 客户端。"""
    def __init__(self, content=None, raise_exc=False, empty=False):
        self._content, self._raise, self._empty = content, raise_exc, empty
        self.chat = type("C", (), {"completions": self})()

    async def create(self, **kw):
        if self._raise:
            raise RuntimeError("boom")
        if self._empty:
            return type("R", (), {"choices": []})()
        return FakeResp(self._content)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_happy_json():
    c = FakeClient('["确立关系","在一起的起点","4.1纪念日"]')
    out = run(expand_query("咱俩怎么开始的", c, "deepseek-chat"))
    assert out[0] == "咱俩怎么开始的", out
    assert "确立关系" in out and "4.1纪念日" in out, out
    assert len(out) == 4, out  # 原 + 3 角度


def test_code_fence():
    c = FakeClient('```json\n["底线承诺","不被污染的约定"]\n```')
    out = run(expand_query("你答应过我啥", c, "m"))
    assert out == ["你答应过我啥", "底线承诺", "不被污染的约定"], out


def test_max_angles_cap():
    c = FakeClient('["a","b","c","d","e"]')
    out = run(expand_query("q", c, "m", config={"max_angles": 2, "min_query_len": 1}))
    assert out == ["q", "a", "b"], out  # 原 + 最多 2


def test_dedup_original():
    c = FakeClient('["q","x"]')  # 角度里混进了原查询
    out = run(expand_query("q", c, "m", config={"min_query_len": 1}))
    assert out == ["q", "x"], out  # 原查询不重复


def test_non_json_fallback():
    c = FakeClient("确立关系, 起点, 纪念日")
    out = run(expand_query("咋开始的", c, "m"))
    assert out[0] == "咋开始的" and "确立关系" in out, out


def test_llm_failure_returns_original():
    c = FakeClient(raise_exc=True)
    out = run(expand_query("重要查询", c, "m"))
    assert out == ["重要查询"], out  # 崩了也不丢原词


def test_empty_choices():
    c = FakeClient(empty=True)
    assert run(expand_query("q", c, "m", config={"min_query_len": 1})) == ["q"]


def test_disabled():
    c = FakeClient('["x"]')
    assert run(expand_query("q", c, "m", config={"enabled": False})) == ["q"]


def test_no_client():
    assert run(expand_query("q", None, "m")) == ["q"]


def test_empty_query():
    assert run(expand_query("   ", FakeClient('["x"]'), "m")) == []


def test_short_query_skipped():
    c = FakeClient('["x"]')
    assert run(expand_query("我", c, "m")) == ["我"]  # 单字 < min_query_len=2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"  ✅ {fn.__name__}")
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: {e}")
        except Exception as e:
            print(f"  💥 {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} 通过")
