"""进程内指标：请求量、各路由分布、token 与成本。

生产环境应换成 Prometheus；这里用内存计数 + 线程锁，够演示也够本地排查。
"""
from __future__ import annotations

import threading
from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.requests = 0
        self.errors = 0
        self.routes: Counter = Counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_cny = 0.0
        self.latencies: list[float] = []

    def record(
        self,
        route: str | None,
        usage: dict | None = None,
        latency_ms: float | None = None,
        error: bool = False,
    ) -> None:
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            self.routes[route or "unknown"] += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                self.cost_cny += float(usage.get("cost_cny", 0.0) or 0.0)
            if latency_ms is not None:
                self.latencies.append(float(latency_ms))
                if len(self.latencies) > 500:
                    self.latencies = self.latencies[-500:]

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies)
            p50 = lat[len(lat) // 2] if lat else 0
            p95 = lat[int(len(lat) * 0.95)] if lat else 0
            return {
                "requests": self.requests,
                "errors": self.errors,
                "routes": dict(self.routes),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "cost_cny": round(self.cost_cny, 6),
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
            }


metrics = Metrics()
