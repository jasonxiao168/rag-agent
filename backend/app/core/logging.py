"""统一日志：结构化、带 trace_id，方便排查一次请求到底走了哪条链路。"""
from __future__ import annotations

import logging
import sys
import uuid

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    _CONFIGURED = True


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
