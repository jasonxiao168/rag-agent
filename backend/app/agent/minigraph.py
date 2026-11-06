"""LangGraph 的最小兼容层。

存在的意义：Lite 环境 / 无外网 CI 里没有 langgraph 时，Agent 依然能按同一张状态图执行。
只实现本项目用到的子集：add_node / add_edge / add_conditional_edges /
set_entry_point / compile / invoke。

优先级：安装了 langgraph 就用真库（见 graph.py），否则自动落到这里。
这本身也是个面试点：编排层与执行层解耦，换框架不改业务节点。
"""
from __future__ import annotations

from typing import Callable

END = "__end__"


class CompiledGraph:
    def __init__(self, graph: "StateGraph") -> None:
        self.graph = graph

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        state = dict(state)
        current = self.graph.entry
        if current is None:
            raise RuntimeError("未设置入口节点")

        steps = 0
        while current != END:
            steps += 1
            if steps > 25:
                raise RuntimeError("状态图执行超过 25 步，疑似死循环")

            node = self.graph.nodes.get(current)
            if node is None:
                raise RuntimeError(f"未知节点：{current}")
            update = node(state) or {}
            state.update(update)

            if current in self.graph.conditional:
                path_fn, mapping = self.graph.conditional[current]
                target = path_fn(state)
                current = (mapping or {}).get(target, target)
            else:
                current = self.graph.edges.get(current, END)

        return state


class StateGraph:
    def __init__(self, schema=None) -> None:
        self.schema = schema
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Callable, dict | None]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        if target != END and target not in self.nodes and source not in self.nodes:
            raise ValueError("节点需先注册")
        self.edges[source] = target

    def add_conditional_edges(self, source: str, path_fn: Callable, path_map: dict | None = None) -> None:
        self.conditional[source] = (path_fn, path_map)

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def compile(self) -> CompiledGraph:
        return CompiledGraph(self)
