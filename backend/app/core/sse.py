"""SSE（Server-Sent Events）协议封装。

事件约定（前端的 api.ts 与这里一一对应）：
  meta    -> 会话/请求元信息
  trace   -> 一个编排步骤（路由、检索、Rerank、工具调用...）
  sources -> 引用来源列表
  token   -> 模型输出的增量文本
  usage   -> 本次请求的 token 与成本
  error   -> 异常
  done    -> 流结束
"""
from __future__ import annotations

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_comment(text: str = "keep-alive") -> str:
    return f": {text}\n\n"
