"""模型客户端：一套接口，两种实现。

- DashScopeLLM：阿里云百炼（OpenAI 兼容协议），真模型；
- MockLLM：无 Key / 无外网时的确定性替身，保证整条链路（分块→检索→Rerank→Agent→评测）
  在 CI 和面试现场都能跑通。

调用方只依赖 `LLMClient` 协议，切换实现不影响业务代码。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from app.config import settings
from app.core.logging import get_logger
from app.core.text import char_tokens, lexical_overlap, tokenize, truncate
from app.core.tokens import BudgetExceeded, Usage, count_messages_tokens, estimate_tokens

logger = get_logger("llm")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    text: str
    usage: Usage
    latency_ms: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        stage: str = "chat",
    ) -> LLMResult: ...

    def stream_chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stage: str = "compose",
        usage_sink: list[Usage] | None = None,
    ) -> Iterator[str]: ...

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        stage: str = "tool_plan",
    ) -> LLMResult: ...

    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]: ...

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> tuple[list[tuple[int, float]], Usage]: ...


def extract_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象，兼容 ```json 代码块与前后废话。"""
    if not text:
        return {}
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(cleaned[start : end + 1])
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# 真模型：DashScope
# --------------------------------------------------------------------------- #
class DashScopeLLM:
    """OpenAI 兼容的聊天客户端。

    同一套代码同时服务 DashScope 与 DeepSeek：差别只在 base_url / key / 模型名。
    `embed()` 与 `rerank()` 走 DashScope 原生协议，仅在 DashScope 作为向量/精排
    后端时使用；DeepSeek 没有这两个接口，由 CompositeLLM 注入本地实现。
    """

    name = "dashscope"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_generation: str | None = None,
        model_fast: str | None = None,
        name: str = "dashscope",
    ) -> None:
        self.api_key = api_key or settings.dashscope_api_key
        self.base_url = base_url or settings.base_url
        self.model_generation = model_generation or settings.model_generation
        self.model_fast = model_fast or settings.model_fast
        self.name = name
        self._client = None

    # -- 内部工具 ---------------------------------------------------------- #
    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "缺少 openai SDK，请先执行 `pip install -r backend/requirements.txt`，"
                    "或设置 MOCK_MODE=1 以 mock 模式运行。"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=settings.request_timeout,
            )
        return self._client

    @staticmethod
    def _usage(resp_usage: Any, model: str, stage: str) -> Usage:
        if resp_usage is None:
            return Usage(model=model, stage=stage)
        return Usage(
            model=model,
            prompt_tokens=getattr(resp_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp_usage, "completion_tokens", 0) or 0,
            stage=stage,
        )

    # -- Chat -------------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        stage: str = "chat",
    ) -> LLMResult:
        model = model or self.model_generation
        kwargs: dict[str, Any] = dict(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                resp = self._openai().chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                return LLMResult(
                    text=text,
                    usage=self._usage(resp.usage, model, stage),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM 调用失败(%s/2): %s", attempt + 1, exc)
                time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败：{last_error}")

    def stream_chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stage: str = "compose",
        usage_sink: list[Usage] | None = None,
    ) -> Iterator[str]:
        model = model or self.model_generation
        client = self._openai()
        collected: list[str] = []
        usage: Usage | None = None

        def _run(include_usage: bool) -> Iterator[str]:
            nonlocal usage
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
            for chunk in client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None):
                    usage = self._usage(chunk.usage, model, stage)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    collected.append(delta)
                    yield delta

        try:
            yield from _run(include_usage=True)
        except TypeError:
            # 部分代理/旧版本不支持 stream_options，退化为本地估算
            collected.clear()
            yield from _run(include_usage=False)

        if usage is None:
            usage = Usage(
                model=model,
                prompt_tokens=count_messages_tokens(messages),
                completion_tokens=estimate_tokens("".join(collected)),
                stage=stage,
            )
        if usage_sink is not None:
            usage_sink.append(usage)

    # -- Function Calling -------------------------------------------------- #
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        stage: str = "tool_plan",
    ) -> LLMResult:
        model = model or self.model_fast
        resp = self._openai().chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=256,
        )
        message = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))
        return LLMResult(
            text=(message.content or "").strip(),
            usage=self._usage(resp.usage, model, stage),
            tool_calls=tool_calls,
        )

    # -- Embedding --------------------------------------------------------- #
    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]:
        client = self._openai()
        vectors: list[list[float]] = []
        prompt_tokens = 0
        batch_size = max(1, settings.embed_batch_size)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            kwargs: dict[str, Any] = dict(model=settings.model_embedding, input=batch)
            if settings.embed_dim:
                kwargs["dimensions"] = settings.embed_dim
            try:
                resp = client.embeddings.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding 带 dimensions 调用失败，重试默认维度：%s", exc)
                resp = client.embeddings.create(model=settings.model_embedding, input=batch)
            vectors.extend([item.embedding for item in resp.data])
            prompt_tokens += getattr(resp.usage, "total_tokens", 0) or 0
        return vectors, Usage(
            model=settings.model_embedding, prompt_tokens=prompt_tokens, stage=stage
        )

    # -- Rerank（DashScope 原生接口，非 OpenAI 协议） ----------------------- #
    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        if not documents:
            return [], Usage(model=settings.model_rerank, stage="rerank")
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少 httpx，无法调用 Rerank 接口") from exc

        url = f"{settings.native_base_url}/api/v1/services/rerank/text-rerank/text-rerank"
        payload = {
            "model": settings.model_rerank,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": top_n},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=settings.request_timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("output", {}).get("results", []) or []
        ranked = [(int(item["index"]), float(item.get("relevance_score", 0.0))) for item in results]
        total_tokens = int(data.get("usage", {}).get("total_tokens", 0) or 0)
        return ranked, Usage(model=settings.model_rerank, prompt_tokens=total_tokens, stage="rerank")


# --------------------------------------------------------------------------- #
# Mock：确定性替身（无网络可用）
# --------------------------------------------------------------------------- #
class MockLLM:
    """用规则模拟模型行为，只为了把工程链路跑通，不代表真实效果。"""

    name = "mock"

    def __init__(self) -> None:
        self._dim = settings.embed_dim or 1024

    # -- Chat -------------------------------------------------------------- #
    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        stage: str = "chat",
    ) -> LLMResult:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        question = messages[-1]["content"] if messages else ""

        if "意图路由器" in system:
            text = self._route(question)
        elif "检索质量评估器" in system:
            text = self._grade(question)
        elif "回答生成器" in system:
            text = self._answer(system, question)
        elif "工具结果整理器" in system:
            text = self._tool_answer(system)
        elif "对话摘要压缩" in system:
            text = truncate(question, 180)
        elif "评测裁判" in system:
            text = self._judge(system)
        elif "工具调用规划" in system:
            text = self._tool_plan(question)
        else:
            text = "我是理然企业智能助手，可以查询内部制度、产品资料、订单物流，也能创建售后工单。"

        return LLMResult(
            text=text,
            usage=Usage(
                model="mock",
                prompt_tokens=count_messages_tokens(messages),
                completion_tokens=estimate_tokens(text),
                stage=stage,
            ),
            latency_ms=0,
        )

    def stream_chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stage: str = "compose",
        usage_sink: list[Usage] | None = None,
    ) -> Iterator[str]:
        result = self.chat(messages, stage=stage)
        # 按 2~4 字切片模拟流式，让前端打字机效果可见
        text = result.text
        step = 3
        for i in range(0, len(text), step):
            yield text[i : i + step]
        if usage_sink is not None:
            usage_sink.append(result.usage)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        model: str | None = None,
        stage: str = "tool_plan",
    ) -> LLMResult:
        question = messages[-1]["content"] if messages else ""
        raw = self._tool_plan(question)
        data = extract_json(raw) or {}
        calls: list[ToolCall] = []
        if data.get("tool"):
            calls.append(
                ToolCall(id="call_mock_1", name=data["tool"], arguments=data.get("arguments", {}))
            )
        return LLMResult(
            text="",
            usage=Usage(
                model="mock",
                prompt_tokens=count_messages_tokens(messages),
                completion_tokens=estimate_tokens(raw),
                stage=stage,
            ),
            tool_calls=calls,
        )

    # -- Embedding --------------------------------------------------------- #
    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]:
        vectors = [self._hash_vector(text) for text in texts]
        usage = Usage(
            model="mock",
            prompt_tokens=sum(estimate_tokens(t) for t in texts),
            stage=stage,
        )
        return vectors, usage

    def _hash_vector(self, text: str) -> list[float]:
        """Hashing Trick：同一 token 永远落到同一维，可做相似度计算。

        真实语义能力有限，但足以验证「索引→召回→排序」的工程正确性。
        """
        vec = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[index] += sign * (1.0 + min(len(token), 6) * 0.15)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # -- Rerank ------------------------------------------------------------ #
    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        query_tokens = tokenize(query)
        scored: list[tuple[int, float]] = []
        for index, doc in enumerate(documents):
            doc_tokens = tokenize(doc)
            overlap = lexical_overlap(query_tokens, doc_tokens)
            # 数字命中额外加权：制度类问答里数字往往就是答案
            numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
            number_hit = len(numbers & set(re.findall(r"\d+(?:\.\d+)?", doc))) / max(len(numbers), 1)
            score = 0.7 * overlap + 0.3 * number_hit
            scored.append((index, round(score, 4)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n], Usage(
            model="mock", prompt_tokens=sum(estimate_tokens(d) for d in documents), stage="rerank"
        )

    # -- 规则实现 ---------------------------------------------------------- #
    @staticmethod
    def _strip_question(raw: str) -> str:
        for marker in ("【用户当前问题】", "【用户问题】", "【对话摘要】"):
            if marker in raw:
                raw = raw.split(marker)[-1]
        return raw.strip()

    def _route(self, raw: str) -> str:
        question = self._strip_question(raw)
        has_order_id = bool(re.search(r"ORD[-_ ]?\d{4}[-_ ]?\d{3,4}", question, re.I))
        tool_words = r"(我的订单|我的包裹|我的快递|我的退款|订单状态|物流到哪|发货了吗|签收|工单)"
        policy_words = r"(政策|规定|制度|标准|时效|规则|流程|多久|怎么申请|能不能退)"
        smalltalk_words = r"^(你好|您好|hi|hello|在吗|谢谢|感谢|你是谁|你能做什么|你能干什么|介绍一下你)"

        if has_order_id or (re.search(tool_words, question) and not re.search(policy_words, question)):
            intent, reason = "tool", "命中订单号或明确的业务数据诉求"
        elif re.search(smalltalk_words, question.strip(), re.I):
            intent, reason = "smalltalk", "寒暄或能力咨询"
        elif len(question) <= 4:
            intent, reason = "clarify", "问题过短，信息不足"
        else:
            intent, reason = "kb", "需要检索内部知识"

        search_query = question if intent == "kb" else ""
        return json.dumps(
            {"intent": intent, "reason": reason, "search_query": search_query}, ensure_ascii=False
        )

    def _grade(self, raw: str) -> str:
        question = self._strip_question(raw)
        blocks = _parse_context_blocks(raw)
        if not blocks:
            return json.dumps({"sufficient": False, "reason": "无检索结果", "rewritten_query": question}, ensure_ascii=False)
        idf, default_idf = _build_idf(blocks)
        best = max(
            _idf_overlap(char_tokens(question), char_tokens(_content_text(b["text"])), idf, default_idf)
            for b in blocks
        )
        sufficient = best >= 0.4
        return json.dumps(
            {
                "sufficient": sufficient,
                "reason": f"最高词面覆盖率 {best:.2f}",
                "rewritten_query": "" if sufficient else f"{question} 制度 标准",
            },
            ensure_ascii=False,
        )

    def _answer(self, system: str, raw_question: str) -> str:
        question = self._strip_question(raw_question)
        blocks = _parse_context_blocks(system)
        if not blocks:
            return "抱歉，我不知道。"

        query_tokens = char_tokens(question)
        # 问「多少/什么/上限/多久」时，带数字的句子更可能是答案
        wants_fact = bool(
            re.search(
                r"(多少|几|多久|多长|上限|标准|金额|浓度|时间|条件|什么|哪|谁|怎么|如何|是否|可以)",
                question,
            )
        )
        # 年份（如 2026）几乎没有区分度，不作为「命中关键数字」的依据
        question_numbers = {
            n
            for n in re.findall(r"\d+(?:\.\d+)?", question)
            if not (len(n) == 4 and n.startswith("20"))
        }
        asks_time = bool(re.search(r"(什么时候|何时|几点|几月|哪一年|日期|生效|开始)", question))
        idf, default_idf = _build_idf(blocks)
        # 中文问句的重心通常在「的」之后：「X 的 Y 是多少」里 Y 才是要找的属性
        focus = question.rsplit("的", 1)[-1] if "的" in question else ""
        focus_chars = set(char_tokens(focus))

        def sentence_score(sentence: str) -> float:
            score = _idf_overlap(query_tokens, char_tokens(sentence), idf, default_idf)
            if question_numbers & set(re.findall(r"\d+(?:\.\d+)?", sentence)):
                score += 0.25  # 问题里的关键数字命中，几乎可以确定是答案句
            elif wants_fact and re.search(r"\d", sentence):
                score += 0.06
            if asks_time and re.search(r"\d{4}\s*年|\d{1,2}\s*月|\d{1,2}\s*日|第[一二三四]季度", sentence):
                score += 0.2
            if focus_chars:
                sentence_chars = set(char_tokens(sentence))
                score += 0.15 * len(focus_chars & sentence_chars) / len(focus_chars)
            return score

        # 先选块：用「块内最高两句之和」，避免表格标题/正文相互稀释
        best_block: dict | None = None
        best_key = 0.0
        best_scored: list[tuple[int, float, str]] = []

        for block in blocks:
            scored = [
                (ordinal, sentence_score(sentence), sentence)
                for ordinal, sentence in enumerate(_content_sentences(block["text"]))
            ]
            # 用「最高单句分」选块：表格/长段落里句子多，用求和会让长块占便宜
            key = max((item[1] for item in scored), default=0.0)
            if key > best_key:
                best_block, best_key, best_scored = block, key, scored

        if not best_block or best_key < 0.34:
            return "抱歉，我不知道。"

        # 证据门槛：中文二元组覆盖率太低时，说明问题里的实体在上下文里根本没出现。
        # 只看单字容易被「公」「司」「日」这类高频字蒙混过关。
        query_bigrams = _cjk_bigrams(question)
        if query_bigrams:
            block_bigrams = _cjk_bigrams(_content_text(best_block["text"]))
            coverage = len(query_bigrams & block_bigrams) / len(query_bigrams)
            if coverage < 0.2 and not (question_numbers & set(re.findall(r"\d+(?:\.\d+)?", best_block["text"]))):
                return "抱歉，我不知道。"

        # 命中表格时整表返回，但要求「最佳答案句本身就是表格行」，否则正文更相关时会被表格抢答
        top_sentence = max(best_scored, key=lambda item: item[1])
        table = _extract_table(best_block["text"])
        if (
            table
            and top_sentence[2].lstrip().startswith("|")
            and sentence_score(table) >= 0.55
            and re.search(r"(多少|几|上限|标准|金额|比例|周期|时长|谁|多久)", question)
        ):
            return f"{table}\n\n[{best_block['index']}]"

        top = sorted([item for item in best_scored if item[1] > 0], key=lambda x: -x[1])[:3]
        top.sort(key=lambda x: x[0])

        parts: list[str] = []
        for _ordinal, _score, sentence in top:
            cleaned = sentence.strip().lstrip("*").strip().strip("|").strip().rstrip("。；;！!？?，,")
            if cleaned:
                parts.append(cleaned)

        if not parts:
            return "抱歉，我不知道。"
        return f"{'；'.join(parts)}[{best_block['index']}]。"

    def _tool_answer(self, system: str) -> str:
        lines = [line for line in system.splitlines() if line.strip().startswith("返回：")]
        if not lines:
            return "抱歉，没有查询到相关业务数据。"
        return "\n".join(line.replace("返回：", "", 1).strip() for line in lines)

    def _tool_plan(self, raw: str) -> str:
        question = self._strip_question(raw)
        order_match = re.search(r"ORD[-_ ]?\d{4}[-_ ]?\d{3,4}", question, re.I)
        order_id = order_match.group(0).replace(" ", "").replace("_", "-").upper() if order_match else ""
        if order_id:
            if re.search(r"(工单|投诉|投诉到|转人工|售后单)", question):
                return json.dumps(
                    {
                        "tool": "create_ticket",
                        "arguments": {"order_id": order_id, "reason": question, "priority": "normal"},
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {"tool": "query_user_order", "arguments": {"order_id": order_id}}, ensure_ascii=False
            )
        return json.dumps({"tool": None, "arguments": {}}, ensure_ascii=False)

    def _judge(self, system: str) -> str:
        def section(name: str, nxt: str | None) -> str:
            if name not in system:
                return ""
            body = system.split(name)[-1]
            if nxt and nxt in body:
                body = body.split(nxt)[0]
            return body.strip()

        gold = section("【参考答案】", "【系统回答】")
        answer = section("【系统回答】", None)
        gold_tokens, answer_tokens = set(tokenize(gold)), set(tokenize(answer))
        if not gold_tokens:
            return json.dumps({"score": 0.0, "correct": False, "hallucination": False, "reason": "参考答案为空"}, ensure_ascii=False)

        overlap = len(gold_tokens & answer_tokens) / len(gold_tokens)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", gold))
        number_hit = len(numbers & set(re.findall(r"\d+(?:\.\d+)?", answer))) / max(len(numbers), 1)
        score = round(0.6 * overlap + 0.4 * number_hit, 2)
        correct = score >= 0.6
        hallucination = (not correct) and ("抱歉" not in answer) and score < 0.3
        return json.dumps(
            {"score": score, "correct": correct, "hallucination": hallucination, "reason": f"mock 词面重合 {overlap:.2f}"},
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
_BLOCK_START = re.compile(r"(?m)^\[(\d+)\] 来源：《(.+?)》(?: > (.+))?$")


def _weighted_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """按 token 长度加权的词面覆盖：越长（越具体）的 token 命中越有价值。"""
    query = set(query_tokens)
    if not query:
        return 0.0
    matched = query & set(doc_tokens)
    total = sum(len(token) for token in query) or 1
    return sum(len(token) for token in matched) / total


def _parse_context_blocks(text: str) -> list[dict]:
    """按块起始行切片，避免正文里的空行/方括号干扰解析。"""
    matches = list(_BLOCK_START.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].split("\n【要求】")[0].strip()
        if body:
            blocks.append(
                {
                    "index": int(match.group(1)),
                    "title": match.group(2),
                    "section": match.group(3) or "",
                    "text": body,
                }
            )
    return blocks


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|(?<=\n)")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and len(s.strip()) >= 4]


def _content_sentences(text: str) -> list[str]:
    """过滤掉文档标题/章节标题这类没有信息量的短行，避免它们污染相似度打分。

    判定：含标点、含表格分隔符，或长度 >= 30 的，视为正文。
    """
    result = []
    for sentence in _sentences(text):
        stripped = sentence.strip()
        if "|" in stripped or re.search(r"[。；;！？!?，,：:]", stripped) or len(stripped) >= 30:
            result.append(stripped)
    return result


def _content_text(text: str) -> str:
    return "；".join(_content_sentences(text))


def _cjk_bigrams(text: str) -> set[str]:
    """连续汉字串的二元组；比单字更能区分「公司」与「创始人生日」。"""
    grams: set[str] = set()
    for run in re.findall(r"[一-鿿]+", text):
        for i in range(len(run) - 1):
            grams.add(run[i : i + 2])
    return grams


def _extract_table(text: str, limit: int = 800) -> str:
    """抽出 Markdown 表格；表格行本身很短，逐行做词面匹配会天然吃亏。"""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return ""
    return "\n".join(lines)[:limit]


def _build_idf(blocks: list[dict]) -> tuple[dict[str, float], float]:
    """在候选上下文内统计 IDF：越稀有的词越能区分是哪个块在回答问题。

    没有语料级 IDF 时，这是一个足够好的近似；也让 mock 不会被「经销商」「公司」
    这类到处都有的词带偏。
    """
    n = max(len(blocks), 1)
    df: dict[str, int] = {}
    for block in blocks:
        for token in set(char_tokens(_content_text(block["text"]))):
            df[token] = df.get(token, 0) + 1
    idf = {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}
    return idf, math.log(n + 1) + 1.0


def _idf_overlap(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], default: float) -> float:
    query = set(query_tokens)
    if not query:
        return 0.0
    matched = query & set(doc_tokens)
    denominator = sum(idf.get(token, default) for token in query) or 1.0
    numerator = sum(idf.get(token, default) for token in matched)
    return numerator / denominator


_LLM: LLMClient | None = None


# --------------------------------------------------------------------------- #
# 向量 / 精排后端（与聊天模型解耦）
# --------------------------------------------------------------------------- #
def _lexical_scores(query: str, documents: list[str]) -> list[float]:
    query_tokens = tokenize(query)
    query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
    scores = []
    for doc in documents:
        overlap = lexical_overlap(query_tokens, tokenize(doc))
        number_bonus = 0.2 if query_numbers & set(re.findall(r"\d+(?:\.\d+)?", doc)) else 0.0
        scores.append(min(1.0, overlap * 0.8 + number_bonus))
    return scores


class LexicalReranker:
    """词面精排：无外部依赖，作为没有 rerank 服务时的兜底。"""

    name = "lexical"

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        if not documents:
            return [], Usage(model="lexical", stage="rerank")
        scores = _lexical_scores(query, documents)
        ranked = sorted(
            ((index, round(score, 4)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        return ranked, Usage(model="lexical", stage="rerank")


def _parse_score_list(text: str, expected: int) -> list[float] | None:
    match = re.search(r"\[[\s\d,.\-]+\]", text or "")
    if not match:
        return None
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list) or len(values) != expected:
        return None
    scores = []
    for value in values:
        try:
            scores.append(max(0.0, min(100.0, float(value))))
        except (TypeError, ValueError):
            return None
    return scores


class LLMReranker:
    """用对话模型做 listwise 精排：一次请求给所有候选打分。

    DeepSeek 没有 rerank 接口，这是不引入本地大模型时的折中：
    比词面重排更懂语义，代价是每次检索多一次 LLM 调用（成本随候选数线性增长）。
    """

    name = "llm-rerank"
    SYSTEM = (
        "你是检索结果重排器。根据问题给每个文档的相关性打分，0 到 100 的整数，"
        "只输出 JSON 数组，长度必须等于文档数量，例如 [85, 12, 60]。不要输出任何解释。"
    )

    def __init__(self, chat: "DashScopeLLM", model: str) -> None:
        self.chat = chat
        self.model = model

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        if not documents:
            return [], Usage(model=self.model, stage="rerank")

        numbered = "\n".join(f"[{index}] {doc[:400]}" for index, doc in enumerate(documents))
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"问题：{query}\n\n文档：\n{numbered}"},
        ]

        usage: Usage | None = None
        scores: list[float] | None = None
        try:
            result = self.chat.chat(
                messages, model=self.model, temperature=0.0, max_tokens=160, stage="rerank"
            )
            usage = result.usage
            scores = _parse_score_list(result.text, len(documents))
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 精排失败，回落词面精排：%s", exc)

        if scores is None:
            ranked, fallback_usage = LexicalReranker().rerank(query, documents, top_n=top_n)
            return ranked, usage or fallback_usage

        ranked = sorted(
            ((index, round(score / 100.0, 4)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        return ranked, usage


class FastEmbedReranker:
    """本地 cross-encoder（可选）。需要下载模型，未下载时自动回落词面精排。"""

    name = "fastembed-rerank"

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        self.model_name = model_name or settings.fastembed_rerank_model
        self.cache_dir = cache_dir or settings.fastembed_cache_dir or None
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            self._model = TextCrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        if not documents:
            return [], Usage(model="fastembed", stage="rerank")
        scores = [float(score) for score in self._load().rerank(query, documents)]
        ranked = sorted(
            ((index, round(score, 4)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        return ranked, Usage(model="fastembed", stage="rerank")


class FastEmbedder:
    """本地 ONNX 向量模型（中文推荐 BAAI/bge-small-zh-v1.5，512 维）。"""

    name = "fastembed"

    def __init__(self, model_name: str | None = None, cache_dir: str | None = None) -> None:
        self.model_name = model_name or settings.fastembed_embed_model
        self.cache_dir = cache_dir or settings.fastembed_cache_dir or None
        self._model = None

    def _load(self):
        if self._model is None:
            # huggingface_hub 在导入后仍会读该变量，这里设置可让镜像源生效
            if settings.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = settings.hf_endpoint
            try:
                from fastembed import TextEmbedding

                kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
                self._model = TextEmbedding(self.model_name, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"本地向量模型加载失败（{self.model_name}）：{exc}\n"
                    "常见原因是容器内无法访问 HuggingFace。请先执行：\n"
                    "  docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com "
                    "backend python -m scripts.download_models\n"
                    "或在宿主机预下载后挂载 ./data/fastembed_cache（见 README「离线/受限网络」）。"
                ) from exc
            logger.info("fastembed 模型已加载：%s", self.model_name)
        return self._model

    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]:
        vectors = [[float(value) for value in vector] for vector in self._load().embed(list(texts))]
        # 本地推理没有 API 费用，token 记 0
        return vectors, Usage(model="fastembed", stage=stage)


class _HashEmbedder:
    """最后兜底：哈希向量（无模型、无网络时保证流程可跑）。"""

    name = "hash"

    def __init__(self) -> None:
        self._mock = MockLLM()

    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]:
        vectors, usage = self._mock.embed(texts, stage=stage)
        usage.model = "hash"
        return vectors, usage


class _ChatEmbedder:
    """复用 DashScope 的 embeddings 接口。"""

    name = "dashscope-embed"

    def __init__(self, chat: "DashScopeLLM") -> None:
        self.chat = chat

    def embed(self, texts: list[str], *, stage: str = "embed") -> tuple[list[list[float]], Usage]:
        return self.chat.embed(texts, stage=stage)


class _ChatReranker:
    name = "dashscope-rerank"

    def __init__(self, chat: "DashScopeLLM") -> None:
        self.chat = chat

    def rerank(
        self, query: str, documents: list[str], *, top_n: int
    ) -> tuple[list[tuple[int, float]], Usage]:
        return self.chat.rerank(query, documents, top_n=top_n)


class CompositeLLM:
    """把「聊天 / 向量 / 精排」三个后端拼成一个统一客户端。

    业务代码只依赖 chat / embed / rerank 三个方法，换任何一家服务都不用改。
    """

    def __init__(self, chat, embedder, reranker) -> None:
        self.chat_backend = chat
        self.embedder = embedder
        self.reranker = reranker
        # manifest 里记录「聊天+向量」，精排不影响向量空间，不参与一致性判断
        self.name = f"{chat.name}+{embedder.name}"

    def describe(self) -> dict:
        return {
            "chat": self.chat_backend.name,
            "embedding": self.embedder.name,
            "rerank": self.reranker.name,
        }

    def chat(self, messages, **kwargs):
        return self.chat_backend.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        return self.chat_backend.stream_chat(messages, **kwargs)

    def chat_with_tools(self, messages, tools, **kwargs):
        return self.chat_backend.chat_with_tools(messages, tools, **kwargs)

    def embed(self, texts, *, stage: str = "embed"):
        return self.embedder.embed(texts, stage=stage)

    def rerank(self, query, documents, *, top_n: int):
        return self.reranker.rerank(query, documents, top_n=top_n)


def _fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _build_embedder(provider: str, chat: "DashScopeLLM"):
    backend = (settings.embedding_backend or "auto").lower()
    if backend in {"hash", "mock"}:
        return _HashEmbedder()
    if backend == "dashscope":
        return _ChatEmbedder(chat)
    if backend == "fastembed" or (backend == "auto" and provider == "deepseek"):
        if _fastembed_available():
            return FastEmbedder()
        logger.warning("fastembed 不可用，向量回落到哈希实现")
        return _HashEmbedder()
    if provider == "dashscope":
        return _ChatEmbedder(chat)
    return _HashEmbedder()


def _build_reranker(provider: str, chat: "DashScopeLLM"):
    backend = (settings.rerank_backend or "auto").lower()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "dashscope":
        return _ChatReranker(chat)
    if backend == "llm":
        return LLMReranker(chat, chat.model_fast)
    if backend == "fastembed":
        if _fastembed_available():
            return FastEmbedReranker()
        logger.warning("fastembed 不可用，精排回落到词面实现")
        return LexicalReranker()
    # auto
    if provider == "dashscope":
        return _ChatReranker(chat)
    return LLMReranker(chat, chat.model_fast)


def get_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        if settings.mock_mode:
            _LLM = MockLLM()
        else:
            provider = settings.provider
            if provider == "deepseek":
                chat = DashScopeLLM(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    model_generation=settings.deepseek_model,
                    model_fast=settings.deepseek_fast_model,
                    name="deepseek",
                )
            else:
                chat = DashScopeLLM(name="dashscope")

            _LLM = CompositeLLM(
                chat=chat,
                embedder=_build_embedder(provider, chat),
                reranker=_build_reranker(provider, chat),
            )
        logger.info(
            "LLM = %s (%s)", _LLM.name, getattr(_LLM, "describe", lambda: {})()
        )
    return _LLM


def reset_llm() -> None:
    """测试用：切换 MOCK_MODE 后重建单例。"""
    global _LLM
    _LLM = None
