# 评测报告

- 生成时间：2026-11-06 05:32
- 运行模式：在线模型 deepseek-chat
- 测试集：`testset.jsonl`，共 123 题
- 检索配置：top_k=10，rerank_top_n=3，hybrid=True

## 总览

| 指标 | 数值 |
| --- | --- |
| 回答正确率 | 100.0% |
| 证据召回率（kb） | 99.0% |
| 上下文精度（kb） | 49.0% |
| 幻觉率（不可回答题） | 0.0% |
| 平均延迟 | 2734 ms |
| 总 token | 413544 |
| 总成本 | ¥0.9136 |

## 分类结果

| 类型 | 题数 | 正确率 | 召回率 |
| --- | --- | --- | --- |
| kb | 101 | 100.0% | 99.0% |
| tool | 6 | 100.0% | 100.0% |
| smalltalk | 3 | 100.0% | — |
| unanswerable | 4 | 100.0% | — |
| acl | 5 | 100.0% | — |
| pii | 4 | 100.0% | — |

## 失败案例与归因

无失败案例。

## 复现

```bash
cd backend
python -m scripts.ingest
python -m scripts.evaluate --compare
```

> 评测集包含 kb / tool / smalltalk / unanswerable / acl / pii 六类问题；`unanswerable` 测幻觉，`acl` 测文档权限，`pii` 测字段级脱敏（无权时不得出现明文）。

> `acl` 用例按 `user` 字段切换身份（guest / alice / bob / carol / dave），其余用例默认以 `carol`（渠道运营）身份运行。
