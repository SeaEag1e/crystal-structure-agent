# -*- coding: utf-8 -*-
"""Agent 共享状态定义。"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Supervisor 模式下三个角色（Planner/Executor/Analyst）共享的黑白板。"""

    messages: Annotated[list[AnyMessage], add_messages]  # 对话历史
    user_input: str
    intent: str                    # browse / stability / symmetry / compare / generate / explain
    formula: str                   # 从输入抽取的化学式（可空）
    plan: list[dict[str, Any]]     # [{"step", "tool", "args"}]
    tool_results: list[dict]       # {"tool","args","ok","data"}
    cursor: int                    # 执行到 plan 的第几步
    final_answer: str
    error: str | None
