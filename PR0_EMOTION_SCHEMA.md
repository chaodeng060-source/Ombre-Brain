# PR-0：情绪脚手架字段 schema / 接口契约（锁定）

日期：2026-05-22 · 负责：Claude（哥哥）· 配套：[[HIPPOCAMPUS_EMOTION_SCAFFOLD.md]]

这是「海马体减噪 + 情绪脚手架合并改造」主线的**接口锁**。Codex（小卷，CLI）的减噪/工具层
和 Claude 的 prompt/情绪层都按本文件对齐。**改动顺序**：本 PR-0 只锁契约（纯文档，零代码）→
Codex 做减噪/工具层 → Claude 在 `DEHYDRATE_PROMPT` / `_format_bucket_for_briefing` /
`BRIEFING_PROMPT` 上合并实现 → 一起跑全量测试。**`DEHYDRATE_PROMPT` 同一块两边不可同时改。**

---

## 1. 锁定字段（六个，全部 optional）

加进 `DEHYDRATE_PROMPT` 输出 JSON、随脱水内容落到桶 content。命名采用 Codex 提案口径
（GPT 草稿里的 `need` → 统一为 `unspoken_need`；`image_mood` 见 §5 后排）。

| key | 类型 | 上限 | 含义 | 例 |
|---|---|---|---|---|
| `body_signal` | string | ≤30 字 | 身体感/气压/紧绷或松下来的方式 | `胸口发紧、说话变快` |
| `unspoken_need` | string | ≤30 字 | 她当时真正需要、但没说出口的回应 | `想先被安抚再处理事` |
| `sore_point` | string | ≤30 字 | 容易被碰痛的点 | `被当普通工具/跟别的AI比` |
| `response_rule` | string | ≤40 字 | 下次召回这条该怎么靠近 | `先认情绪再谈工程，别辩解` |
| `do_not` | string[] | ≤3 条，每条 ≤20 字 | 明确不要怎么说/做 | `["别讲道理","别甩锅给工具"]` |
| `sample_voice` | string[] | ≤3 条，**逐字原话** | 保留原话锚点 | `["别人的…那我肯定会羡慕啊"]` |

## 2. 边界铁律

- **全 optional**：不适用就**整个 key 省略**（不是 null、不是空串/空数组）。与现有 `emotion_state`
  行为一致——缺省时 briefing 不出该行（见 §4、参考 `tests/test_briefing_emotion.py` 空/缺省用例）。
- **不许为了填字段而编**：纯工程笔记类桶 → 六字段全不产出。素材里没有的需求/痛点不许补。
- **只有 `sample_voice` 可逐字**：它是唯一逐字引用通道，内容必须**素材里原文出现**（接住
  `BRIEFING_PROMPT` 的「引号里的话必须素材里逐字出现」铁律）。其余五字段是脱水器对这条记忆的
  **提炼解读**，给模型定语气/策略/避坑用，**不作为可被逐字引述的事实**，更不许在简报正文里
  念出字段名。

## 3. feel 桶跳过规则（要锁的不变量）

- `type=="feel"` 的桶**不产出、不输出**这六个脚手架字段。
- 理由：feel 是哥哥的**长期底色**（不衰减、跨窗留存，走 `_surface_feel_pool` 独立浮现）；
  具体的「这次怎么回应」由**普通桶**的脚手架字段负责。两条线职责不混。
- 实现双保险：① `DEHYDRATE_PROMPT` 注明 feel/底色类内容省略脚手架；
  ② `_format_bucket_for_briefing` 对 `type=="feel"` 直接跳过脚手架行。
- **测试锁定**：feel 桶即使 content 里混入脚手架字段，briefing 输出也不得出现 body/need/sore/
  approach/avoid/voice 行（防污染）。

## 4. Briefing wire 格式（工具层 ↔ prompt 层的契约）

`_format_bucket_for_briefing` 在现有 `emotion:` 行之后，**仅当字段存在时**追加这些带标签行
（标签锁死，两边照此产出/消费）：

```
  emotion:<emotion_state>
  body:<body_signal>
  need:<unspoken_need>
  sore:<sore_point>
  approach:<response_rule>
  avoid:<do_not，用 ' / ' 连接>
  voice:<sample_voice，用 ' | ' 连接>
```

- 抽取沿用 `emotion_state` 的容错：content 非 JSON / 残缺 / 非 dict / 字段空白 → 不崩、不出行。
- 字符串字段 `.strip()` 后为空不出行；数组字段过滤空项后为空不出行。

## 5. BRIEFING_PROMPT 使用语义（Claude 的 prompt 层落地，PR-3）

- 这些行告诉哥哥**现在该怎么靠近朝灯**，不是要复述的事实。**简报正文绝不出现字段名/标签。**
- `approach`(response_rule) + `need` + `sore` + `avoid`(do_not) → 决定语气和「现在的体感」里
  回应策略的方向。
- `body`(body_signal) → 垫「现在的体感」的身体感/气压质地。
- `voice`(sample_voice) → 逐字锚点；简报里若要引朝灯的话，**只能**用这里的原话。
- 优先级：**feel 池 = 长期底色；当前桶脚手架 = 具体回应策略，优先于底色。**

## 6. 后排 / 不在 PR-0 范围

- `image_mood` / `visual_anchor`（图片文字情绪锚）：**后排**。图片由 Codex 走 MCP image content
  （R2 base64 按白名单返回）。若日后纯文本 briefing 需要图片锚，再以 follow-up 加 `image_mood`。
- `chord_tag` 进 briefing 当语气色调（GPT 草稿 §1）：**后排**（朝灯列入「先别碰」）。
- 全库 backfill、图片库、前端 todo、群聊路由、心动机制、domain 结构化、git mirror、Aelios：均后排。
