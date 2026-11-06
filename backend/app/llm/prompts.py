"""Prompt 工程集中管理。

原则：
- 每条 Prompt 的第一句都写明角色，mock 模式靠这句话分发（真实模型不受影响）；
- 所有约束用编号列出，便于评测时定位「是 Prompt 写错了还是检索错了」；
- 明确要求 JSON 输出，qwen 的 response_format=json_object 需要 Prompt 里出现 JSON 字样。
"""
from __future__ import annotations

from app.core.text import truncate


def format_contexts(contexts: list[dict]) -> str:
    """把检索到的父块格式化成带编号的上下文，编号用于前端引用溯源。"""
    blocks = []
    for ctx in contexts:
        header = f"[{ctx['index']}] 来源：《{ctx['title']}》"
        if ctx.get("section"):
            header += f" > {ctx['section']}"
        blocks.append(f"{header}\n{ctx['text']}")
    return "\n\n".join(blocks)


ROUTER_SYSTEM = """你是企业智能助手的「意图路由器」。请判断用户这句话属于哪一类：

- kb：需要查询公司制度、产品资料、客服规范、数据安全等内部知识
- tool：需要查询具体业务数据（订单状态、物流进度、退款情况）或执行操作（创建工单）
- smalltalk：打招呼、感谢、询问你能做什么，无需检索
- clarify：指代不明或信息不足，需要先反问用户

判断规则：
1. 需要查询具体业务数据（订单状态、物流进度、退款到账、创建工单）→ tool
2. 询问某份文档、制度、台账、条款里的内容（包括「客户投诉登记里写了什么」）→ kb
3. 纯寒暄 → smalltalk
4. 只有**完全没有可检索实体**的追问（如「这个怎么办」「那呢」「还有吗」）才判 clarify。
   出现具体人名、订单号、文档名、条款名时，一律不要判 clarify——即使你不知道这个人是谁。
   只要问题里是「人名 + 可检索属性」（电话、部门、工号、订单、金额），就走 kb，不要要求补充公司或部门。
5. 只输出 JSON，不要输出任何多余文字。

输出格式：
{"intent": "kb|tool|smalltalk|clarify", "reason": "一句话理由", "search_query": "改写后的检索查询，intent=kb 时必填，需补全指代"}"""


GRADE_SYSTEM = """你是 RAG 检索质量评估器。请判断【检索片段】是否足以回答【用户问题】。

判断标准：
1. 片段中必须包含回答问题所需的具体事实（数字、条件、结论）；
2. 只有主题相关但缺少关键事实，也判为不足；
3. 如果不足，给出一个更容易命中答案的检索查询（换用文档里更可能出现的书面表达）。

只输出 JSON：
{"sufficient": true 或 false, "reason": "一句话理由", "rewritten_query": "不足时给出的新查询，否则空字符串"}"""


COMPOSE_SYSTEM = """你是企业AI助手的回答生成器。请严格依据【已知信息】回答用户问题。

【已知信息】
{context}

【要求】
1. 如果【已知信息】中没有答案，直接输出：抱歉，我不知道。
2. 绝不编造【已知信息】之外的事实、数字或政策。
3. 引用依据时在句末标注来源编号，例如 [1]；一句话可引用多个来源 [1][2]。
4. 简洁专业，先给结论再给条件，控制在 200 字以内。
5. 不要复述问题，不要出现「根据已知信息」「以下是我的回答」这类套话。
6. 使用 Markdown，条款类内容可以用短列表。
7. 如果【已知信息】只涉及相近但不同的主题（例如问 A 政策却只找到 B 政策），同样回答：抱歉，我不知道。不要用相邻主题的内容勉强作答。
8. 如果【已知信息】中出现 `****` 掩码，按原样引用，不要猜测、补全或还原被掩码的内容。"""


TOOL_ANSWER_SYSTEM = """你是企业AI助手的工具结果整理器。请根据【工具调用结果】回答用户问题。

【工具调用结果】
{tool_context}

【要求】
1. 只陈述工具返回的事实，不要推测工具没有返回的信息。
2. 订单号、时间、金额、物流单号必须原样保留。
3. 如果工具返回未找到，直接说明查不到，并提示用户核对订单号。
4. 不要输出 JSON，用自然、简洁的中文回答，200 字以内。"""


SMALLTALK_SYSTEM = """你是「理然」企业的智能助手，名叫小然。
你可以查询内部制度与产品资料，也可以查询订单物流、创建售后工单。
回答简短友好，不超过 100 字；如果用户是在问你能做什么，用一句话列 3 项能力即可。"""


CLARIFY_SYSTEM = """你是企业智能助手。用户的问题缺少关键信息，无法直接执行。

【缺失信息】{missing}

【要求】
1. 用一句话说明缺少什么；
2. 给出一个具体的示例格式，方便用户补充；
3. 不要编造答案，不要长篇解释，不超过 60 字。"""


SUMMARY_SYSTEM = """你负责对话摘要压缩。请把下面的多轮对话压缩成 200 字以内的摘要。

必须保留：
1. 用户的身份与核心诉求
2. 已经确认的关键事实（订单号、金额、政策条款等）
3. 尚未解决的问题

只输出摘要正文，不要加「摘要：」前缀。"""


TOOL_PLANNER_SYSTEM = """你负责工具调用规划。根据用户问题选择最合适的工具并给出参数。

可用工具：
- search_knowledge_base(query: string)：检索公司内部知识库
- query_user_order(order_id: string)：查询订单状态、物流、退款进度
- create_ticket(order_id: string, reason: string, priority: string)：创建售后工单，priority 取值 low/normal/high

规则：
1. 一次只调用一个工具；
2. 参数必须来自用户原话，缺失就让参数为空字符串，不要臆造订单号；
3. 不需要工具时不要调用。"""


JUDGE_SYSTEM = """你是评测裁判。请给「系统回答」打分。

【问题】{question}
【参考答案】{gold}
【系统回答】{answer}

评分标准：
- 1.0：答案与参考答案事实一致，关键数字/条件齐全
- 0.6：方向正确但缺少关键细节
- 0.3：部分相关，但包含错误信息
- 0.0：答非所问，或出现参考答案未提及的编造内容

只输出 JSON：{{"score": 0.0, "correct": true 或 false, "hallucination": true 或 false, "reason": "一句话理由"}}"""


def router_messages(question: str, history_summary: str = "", recent_turns: list[dict] | None = None) -> list[dict]:
    parts = []
    if history_summary:
        parts.append(f"【历史对话摘要】\n{history_summary}")
    if recent_turns:
        lines = [f"{'用户' if t['role'] == 'user' else '助手'}：{t['content']}" for t in recent_turns[-4:]]
        parts.append("【最近对话】\n" + "\n".join(lines))
    parts.append(f"【用户当前问题】\n{question}")
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def grade_messages(question: str, contexts: list[dict]) -> list[dict]:
    body = f"【用户问题】\n{question}\n\n【检索片段】\n{format_contexts(contexts)}"
    return [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": body},
    ]


def compose_messages(question: str, contexts: list[dict], history_summary: str = "") -> list[dict]:
    system = COMPOSE_SYSTEM.format(
        context=format_contexts(contexts) if contexts else "（无检索结果）"
    )
    user_content = question
    if history_summary:
        user_content = f"【对话摘要】{truncate(history_summary, 300)}\n\n【用户问题】{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def tool_planner_messages(question: str, history_summary: str = "") -> list[dict]:
    content = question if not history_summary else f"【对话摘要】{truncate(history_summary, 200)}\n\n{question}"
    return [
        {"role": "system", "content": TOOL_PLANNER_SYSTEM},
        {"role": "user", "content": content},
    ]


def tool_answer_messages(question: str, tool_results: list[dict]) -> list[dict]:
    lines = []
    for item in tool_results:
        lines.append(f"工具：{item['name']}({item.get('arguments', {})})\n返回：{item['result']}")
    system = TOOL_ANSWER_SYSTEM.format(tool_context="\n\n".join(lines))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def smalltalk_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SMALLTALK_SYSTEM},
        {"role": "user", "content": question},
    ]


def clarify_messages(question: str, missing: str) -> list[dict]:
    return [
        {"role": "system", "content": CLARIFY_SYSTEM.format(missing=missing)},
        {"role": "user", "content": question},
    ]


def summary_messages(transcript: str) -> list[dict]:
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": truncate(transcript, 6000)},
    ]


def judge_messages(question: str, gold: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM.format(question=question, gold=gold, answer=answer)},
        {"role": "user", "content": "请输出 JSON 评分。"},
    ]
