# 章节元数据与可核验引用

这一阶段解决的不是“让模型看起来更像有依据”，而是让用户能够从回答里的事实回到
模型实际看到的原文。链路中的编号只有一个来源，避免回答、接口和界面各自维护一套
顺序后发生错位。

## 一、章节如何进入索引

`src/loader.py` 先按段落识别常见中文章节标题，再在每个章节内部切分：

- 支持“第一章”“第十卷”“序章”“楔子”“番外”等常见形式；
- 章节标题会写入该章节每个片段的 `chapter_title`；
- 片段不会跨过章节边界，`chunk_id` 仍是一本书内连续的全局编号；
- 没有章节标题的普通文本继续正常切分，`chapter_title` 为 `NULL`。

识别规则有意保持保守。例如“第一章的内容很精彩。”带有句末标点、并且是普通
陈述句，不会被当作标题。规则测试在 `tests/backend/test_loader.py`。

PostgreSQL 的 `novel_chunks` 表新增了可空字段：

```sql
chapter_title TEXT
```

后端启动时会用 `ADD COLUMN IF NOT EXISTS` 兼容旧表，因此旧服务不会因为缺列直接
启动失败；但旧片段本身没有章节信息，升级后仍需运行一次 `python src/ingest.py`
重建索引，才能填充章节标题。

## 二、引用编号为什么以 prompt 为准

检索结果经过重排后还会补相邻片段。模型最终看到的是“扩展后的上下文”，所以 SSE
接口返回的 `sources` 也必须使用同一份扩展结果，不能返回扩展前的 top-k。

`src/rag.py` 在 prompt 中按最终顺序写入：

```text
[1] 《书名》 · 章节名 · 片段 #123
原文……
```

系统同时要求模型：关键事实后紧跟 `[n]`、只能引用实际存在的编号、多个片段共同
支持时可写 `[1][3]`，不要另写一个无法对应正文的脚注列表。

前端只把有效范围内的编号变成按钮。点击 `[1]` 会滚动到第一个出处卡片并高亮；
模型若输出了不存在的 `[99]`，界面保留原文字样，不会错误链接到别的片段。历史记录
没有 `chapter_title` 时也能正常显示。

## 三、评测能证明什么

`src/citation_eval.py` 和 `scripts/eval_citations.py` 提供三类自动指标：

1. 回答是否包含引用；
2. 引用编号是否落在实际来源范围内；
3. 预期证据关键词是否出现在被引用片段中。

运行已有问答结果：

```bash
python scripts/eval_citations.py tests/results_3b.json
```

新跑问答评测时，`tests/run_qa_tests.py` 会固定使用 `grounded` 模式，并把完整来源先
用于计算引用指标，再截取短文本写入结果文件。

这些规则能发现“没引用”“编号越界”“引用片段缺少预期证据”，但不能证明一句自然
语言结论一定被原文语义蕴含。因此结果保留 `manual_support_review_required=true`，
把“编号格式正确”和“证据真正支持结论”明确分开。后续可以在固定人工标注集上增加
NLI 或 LLM-as-judge，但不能拿自动判断替代抽样复核。

## 四、一次完整验证

```bash
source venv/bin/activate
python src/ingest.py
pytest -q
python scripts/eval_routing.py
cd frontend
npm run build
npm run test:e2e
```

验收时至少检查一个回答：正文出现 `[1]`，点击后滚动到带有正确书名、章节名和片段
编号的原文卡片，引用编号与模型 prompt 中的编号一致。
