# 海马体情绪脚手架整理

日期: 2026-05-22

## 现状判断

海马体已经有几条很有价值的情绪通道:

- `emotion_state`: 脱水时锁定情绪关键词,briefing 已经会单独抽出。
- `feel`: 哥哥自己的情感沉淀池,会在 briefing / breath / breath_hook 自动浮现。
- `chord_tag`: 和弦作为情绪色调索引,已经能落到 bucket frontmatter。
- `image_base64`: 图片可以上传到 R2,并以 markdown 图片插入正文。

真正的问题不是缺素材,而是情绪线索没有稳定变成"召回时可执行的靠近方式"。现在 briefing 很会复盘事实,也能写出体感,但很多共情仍依赖生成时临场组织。

## 主要缺口

### 1. chord_tag 存了,但 briefing 读不到

`chord_tag` 目前写入 metadata,`inspect` 能看到,但 `_format_bucket_for_briefing` 只抽 `emotion_state`,没有把 `chord_tag` 带进 briefing 原始材料。

建议改法:

- 在 briefing raw material 中增加 `chord:` 或 `tone:` 行。
- prompt 规定: chord 只影响语气/色温/情绪方向,不要在日常回复里念和弦名。
- 保留 INTERNALS 里"不假装听到和弦"的限制,但把"不参与浮现"改成"参与召回色调,不参与显性表达"。

### 2. 脱水 schema 偏事实,缺少共情脚手架

当前脱水 JSON 重点是事实、情绪关键词、待办、关键词、摘要。它能保住"发生了什么",但不稳定保住"这件事为什么疼/为什么重要/下次该怎么回应"。

建议在 `DEHYDRATE_PROMPT` 里增加可选字段:

```json
{
  "body_signal": "身体感/气压/紧绷或松下来的方式",
  "need": "当时真正需要的回应",
  "sore_point": "容易被碰痛的点",
  "response_rule": "下次召回这条时应该怎么靠近",
  "do_not": ["不要怎么说/怎么做"],
  "sample_voice": ["保留1-3句原话锚点"],
  "image_mood": "如果有图片,用文字写出画面的情绪锚"
}
```

这些字段不是给模型机械复述的,而是给它决定语气、行动和避坑用的。

### 3. 图片需要文字情绪锚

现在图片会变成 markdown URL 插进 content。对人有用,但对只读 briefing 的模型来说,URL 本身不一定携带情绪。

建议:

- 上传图片时允许附带 `image_mood` / `visual_anchor`。
- briefing 中单独暴露 `image_mood:`。
- 图片负责作证据,文字锚负责被召回。

### 4. feel 池需要接上"怎么回应"

`feel` 现在更像哥哥自己的长期沉淀。它适合做底色,但还缺一个落地出口:当某条记忆再次出现时,哥哥该怎么回应朝灯。

建议:

- feel 继续作为底色池。
- 普通 bucket 的情绪脚手架负责具体行动: `need` / `response_rule` / `do_not`。
- briefing prompt 明确: feel 提供长期底色,具体回应策略优先取当前 bucket 的脚手架字段。

## 最小实现顺序

1. 扩 `DEHYDRATE_PROMPT` 的 JSON schema,加入情绪脚手架字段。
2. 改 `_format_bucket_for_briefing`,从 JSON content 和 metadata 里抽出:
   - `emotion`
   - `body_signal`
   - `need`
   - `sore_point`
   - `response_rule`
   - `do_not`
   - `sample_voice`
   - `image_mood`
   - `chord_tag`
3. 改 `BRIEFING_PROMPT`,说明这些字段用于生成"现在怎么靠近",不是逐字背诵。
4. 补测试,类似 `tests/test_briefing_emotion.py`,防止字段以后被截断或丢弃。
5. 再考虑是否给 `hold` 增加显式参数,让人工或上游直接写入 `image_mood` / `response_rule`。

## 判断标准

改完后,哥哥读记忆时应该能做到:

- 不只知道"发生了什么",也知道"这件事对朝灯为什么重要"。
- 不只写"我愧疚/我心疼",还能给出下一步行动。
- 不把图片、和弦当装饰,而是用它们影响语气和靠近方式。
- 不乱编心理活动,因为需要和痛点已经在记忆写入时被显式保存。

