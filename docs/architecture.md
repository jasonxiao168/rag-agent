# 架构说明

## 1. 分层

```
┌──────────────────────────────────────────────────────────────┐
│ 表现层   React + Vite：对话、引用、执行链路可视化              │
├──────────────────────────────────────────────────────────────┤
│ 接口层   FastAPI：/api/chat（SSE）、health、metrics、reindex  │
├──────────────────────────────────────────────────────────────┤
│ 编排层   LangGraph：router → retrieve → grade → rewrite → …   │
│          （缺失 langgraph 时自动使用 app/agent/minigraph.py）  │
├──────────────────────────────────────────────────────────────┤
│ 能力层   RAG（混合检索 + Rerank） / Tools（Function Calling）  │
├──────────────────────────────────────────────────────────────┤
│ 模型层   聊天（DeepSeek/百炼） + 向量（fastembed/百炼） + 精排（LLM/百炼）│
│          MockLLM：无 Key 时的确定性替身                        │
├──────────────────────────────────────────────────────────────┤
│ 存储层   向量库（Qdrant / numpy） + 父块表 + BM25 索引 + 会话   │
└──────────────────────────────────────────────────────────────┘
```

## 2. 摄入数据流

```
data/raw/*.md|pdf|docx
   │  loader：解析 frontmatter（acl / department），计算 content_hash（正文+ACL）
   ▼
LoadedDoc(doc_id, title, text, acl, content_hash)
   │  build(incremental=True)：与 docs.json 对比 → 新增 / 修改 / 删除 / 未变
   │  只有新增与修改会继续往下走；删除的文档按 child_id 从向量库清除
   ▼
sections[(section_path, body)]
   │  _merge_sections：相邻小节合并到 parent_size（默认 900 字）
   ▼
parents
   │  _windows：按句边界打包，整句重叠；表格作为原子单元
   ▼
children（默认 220 字，重叠 40）
   │  embed_text = "文档标题｜章节\n子块正文"
   ▼
向量库 + children.jsonl + parents.json + docs.json + manifest.json
```

关键文件：

| 文件 | 内容 |
| --- | --- |
| `data/index/children.jsonl` | 约 62 个子块：child_id / parent_id / 正文 / 入库文本 / acl |
| `data/index/parents.json` | 约 17 个父块：回溯用的完整上下文 |
| `data/index/qdrant/` | 嵌入式 Qdrant 的存储目录（设置 `QDRANT_URL` 时改用服务端） |
| `data/index/vectors.npy` | 本地 numpy 后端的向量矩阵（回落时使用） |
| `data/index/manifest.json` | 构建时间、provider、维度、分块参数、本次增量统计 |
| `data/index/docs.json` | 文档清单：ACL、部门、content_hash、块数（增量对比依据） |
| `data/sessions/*.json` | 会话摘要与最近消息（绑定 user_id） |
| `data/tickets.json` | Agent 创建的售后工单 |

manifest 里的 `provider` 用于发现「索引与当前模型不匹配」：MOCK 建索引后换成真实 Key，向量空间不同，必须重建。

## 3. 查询数据流

```
question + user(departments)
  ├─ memory：摘要 + 最近 N 轮（会话与 user_id 绑定）
  ├─ ACL：allowed_docs = {doc | doc.acl 含 all 或与 departments 相交}
  ├─ router（fast 模型，JSON）：intent / reason / search_query
  │
  ├─ kb：embed(query) → 向量 Top-K（allowed_docs 过滤）┐
  │      BM25(query)     → Top-K（allowed_docs 过滤）  ├→ RRF → Top10
  │                                                    └→ Rerank → Top6 子块
  │                                                         → 父块去重（二次校验 ACL）
  │                                                         → Top3 → 字符预算
  │                                                         → PII 脱敏（无 pii 权限时打码）
  │                                                         → grade（CRAG）→ 不足则 rewrite 重试
  │
  ├─ tool：chat_with_tools → 执行器（同样透传 departments）→ 结果回填
  └─ smalltalk / clarify：直接组装
        ↓
     prepare：生成 messages
        ↓
     stream_chat（主模型）→ SSE token
```

## 4. Agent 状态字段

| 字段 | 说明 |
| --- | --- |
| `user_id` / `user_name` / `departments` | 身份与部门，决定 ACL 过滤范围 |
| `acl` | 本次检索的可见文档统计（visible / total） |
| `route` / `route_reason` | 路由结果与理由 |
| `answer_style` | kb / tool / smalltalk / clarify，决定 Prompt |
| `search_query` | 当前检索词（可被改写） |
| `contexts` / `sources` | 父块上下文与引用 |
| `candidates` | 候选子块及 dense/bm25/rrf/rerank 明细 |
| `grade_sufficient` / `rewritten_query` / `retries` | CRAG 状态 |
| `tool_results` | 工具名、参数、返回、耗时 |
| `gen_messages` | 最终 Prompt |
| `trace` | 给前端的链路日志 |
| `usages` | 每个阶段的 token 与成本 |

## 5. 降级矩阵

| 依赖 | 缺失时的行为 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 进入 MockLLM：路由用关键词、回答用抽取、embedding 用哈希 |
| `qdrant-client` | 回落 `NumpyVectorStore`（.npy + jsonl） |
| `langgraph` | 回落 `minigraph`，节点与条件边不变 |
| `jieba` | 分词退化为「汉字 + 二元组 + 英文词」 |
| `pypdf` / `python-docx` | 跳过对应格式并打日志 |
| 索引不存在 | `/api/health` 返回 degraded，检索节点给明确提示 |

## 6. 参数与调优

| 参数 | 默认 | 影响 |
| --- | --- | --- |
| `PARENT_SIZE` | 900 | 父块越大，上下文越全，token 越多 |
| `CHILD_SIZE` | 220 | 子块越小，召回越准，索引越大 |
| `RETRIEVE_TOP_K` | 10 | 粗排数量，影响 Rerank 成本 |
| `RERANK_TOP_N` | 3 | 进入 Prompt 的父块数 |
| `CONTEXT_MAX_CHARS` | 3200 | 上下文硬上限 |
| `GRADE_HIGH_SCORE` | 0.50 | 高于此值跳过 LLM 判定 |
| `GRADE_LOW_SCORE` | 0.12 | 低于此值直接判不足 |
| `MAX_HISTORY_TURNS` | 4 | 滑动窗口保留轮数 |

调参顺序建议：先看评测报告的失败归因。检索未命中优先调 `CHILD_SIZE` 与 BM25 分词；上下文合理但事实缺失优先调 `RERANK_TOP_N`；未拒答优先收紧 Prompt 与阈值。

## 7. 部署拓扑

```
浏览器 ──▶ nginx（:8080）
             ├── /*        → 静态文件（React 构建产物）
             └── /api/*    → FastAPI（:8000，关闭 proxy_buffering）
                               ├── Qdrant（Docker: http://qdrant:6333）
                               │          本地: 嵌入式 data/index/qdrant
                               └── data/（volume：语料、父块表、会话）
```

SSE 必须关闭 nginx 缓冲（`proxy_buffering off` + `X-Accel-Buffering: no`），否则会表现为「卡住后一次性刷出」。

## 8. 权限与增量索引（实现细节）

### 8.1 权限模型

- 文档级 ACL，粒度是「部门」；`all` 表示全员。
- 生效点：向量检索、BM25、父块回溯、工具调用、会话隔离。任何一处漏掉都可能成为越权入口。
- 返回值只带可见文档；trace 里只暴露「可见 N/M 篇」这种聚合数字，不暴露被过滤文档的标题。
- 无权问题的正确行为是拒答，而不是用相邻主题的可见文档硬答。评测集 `acl-*` 用例专门测这一点。
- 字段级脱敏是第二道：文档可见，但 `permissions` 不含 `pii` 时，手机号/身份证/银行卡/邮箱在进入 Prompt 前被打码；引用与候选预览同样只保留掩码。

### 8.2 增量构建与原子切换

```
docs = load_documents(raw)
old = docs.json
added   = 新 doc_id
updated = content_hash 不同
removed = old 中存在、docs 中不存在
unchanged = 其余

for doc in updated + removed:
    store.delete(child_ids_of(doc))
    drop children/parents/docs entries

embed(added + updated)
store.add(...)
persist()
```

`content_hash` 覆盖正文、ACL 与归属部门，所以改权限也会触发重建。

`rebuild_atomic()` 先把整个索引复制到 `index.staging`，在 staging 上执行增量构建，成功后：

```
index            →  index.old
index.staging    →  index
rm -rf index.old
```

构建失败则删除 staging，正式索引完全不受影响。目录 rename 在同一文件系统内是原子的；多副本部署仍建议用版本化 collection + 别名切换。

`rebuild_atomic()` 会自动识别后端：本地 numpy 走 staging + rename；Qdrant 的向量存储在自己的集合/目录中，复制索引目录并不能复制向量，因此退化为原地增量重建并在 manifest 标记 `swap.atomic=false`。生产若要求原子切换，用版本化 collection + alias。

### 8.3 多轮对话

- 会话按 `session_id + user_id` 隔离；
- 问题以「那 / 它 / 还有」开头且模型没有改写 query 时，用上一轮用户问题拼接后再路由与检索；
- 工具规划同样使用补全后的问题，否则「它什么时候到」里没有订单号；
- `evaluate_multiturn.py` 覆盖 6 组指代用例，并检查跨轮脱敏是否一致。

### 8.5 向量库：Qdrant

| 形态 | 配置 | 特点 |
| --- | --- | --- |
| 嵌入式 | 不设 `QDRANT_URL` | 零运维，数据在 `data/index/qdrant`；同一目录只允许一个客户端，重建前必须先 `close()` |
| 服务端 | `QDRANT_URL=http://qdrant:6333` | 支持并发与横向扩展；`docker-compose` 已内置 qdrant 服务 |

- point id 用 `UUID5(child_id)` 稳定映射，payload 里保留原始 `child_id` 与文档元数据，检索结果能直接回溯父块；
- ACL 用 payload 过滤（`doc_id` 的 `MatchAny`），在 HNSW 检索时生效，不是取回后再删；
- 增量删除按 point id 精确删除，不重建 collection；`reset()` 只在全量重建时调用。

### 8.6 已知边界

- 改一个字也会整篇重算（文档级 hash）；
- 原子切换依赖目录 rename，Windows 上需额外处理；多进程部署建议版本化 collection；
- 权限只到文档级 + PII 字段级，通用的任意字段策略（如按金额区间）未实现。
