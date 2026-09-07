# -*- coding: utf-8 -*-
"""LangGraph 图编排：意图识别 → 任务规划 → 工具调度循环 → 结果汇总。

对应三个 Agent 角色（Supervisor 模式）：
- Planner Agent（planner_node）：LLM 产出结构化任务计划
- Executor Agent（executor_node）：逐步调度工具，条件边循环
- Analyst Agent（summarizer_node）：汇总工具结果为自然语言报告
"""
from __future__ import annotations

import json
import re
from typing import Any, Generator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agent.llm import get_llm, has_api_key
from agent.prompts import (
    INTENT_PROMPT,
    PLANNER_PROMPT,
    SUMMARIZER_PROMPT,
    SYSTEM_PROMPT,
)
from agent.state import AgentState
from agent.tools import TOOL_MAP
from core.structures import list_formulas, load_index

_VALID_INTENTS = {"browse", "stability", "symmetry", "compare", "generate", "explain"}


# ---------- JSON 抽取助手（GLM-4-Flash 偶发 markdown 包裹，统一正则处理） ----------

def _extract_json(text: str):
    """从 LLM 输出中抽取 JSON（数组或对象），失败返回 None。"""
    text = text.strip()
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ---------- 规则兜底（LLM 不可用/解析失败时不挂） ----------

_INTENT_KEYWORDS = [
    # 顺序即优先级：强信号词先匹配；规则模式可组合多个意图
    ("compare", ("对比", "比较", "和实验", "相对照", "compare", "experiment", "reference", " vs ")),
    ("stability", ("稳定", "形成能", "排序", "排名", "stable", "formation", "rank")),
    ("symmetry", ("空间群", "对称", "极性", "铁电", "space group", "symmetry", "polar")),
    ("generate", ("帮我生成", "生成新", "生成几个", "create new", "generate new")),
    ("browse", ("浏览", "列出", "查询", "看看", "list", "browse", "show", "find")),
]

_FORMULA_RE = re.compile(
    r"\b([A-Z][a-z]?(?:P(?:S|Se)3(?:I|Br|Cl))?)\b"
)


def _rule_intent(text: str) -> tuple[list[str], str | None]:
    """规则意图识别：返回全部命中意图（组合规划用）+ 化学式。"""
    lowered = text.lower()
    intents = [intent for intent, kws in _INTENT_KEYWORDS
               if any(k in lowered for k in kws)]
    m = _FORMULA_RE.search(text)
    return (intents or ["browse"]), (m.group(1) if m else None)


def _rule_plan(intents: list[str], formula: str | None) -> list[dict]:
    """组合规划：多意图 → 链式工具计划（query 只出现一次，后续步骤引用其输出）。"""
    f = formula or ""
    if "generate" in intents:
        return [{"step": 1, "tool": "generate_local_gpu",
                 "args": {"formula": f or "GaPS3I", "num_structures": 3}}]
    if intents == ["explain"]:
        return []

    steps = [{"step": 1, "tool": "query_structure_library",
              "args": {"formula": f, "limit": 20 if "stability" in intents else 8}}]
    ref, n = "query_structure_library", 2
    if "stability" in intents:
        steps.append({"step": n, "tool": "filter_stable",
                      "args": {"candidates": {"$from": ref},
                               "threshold": 0.0, "top_k": 10}})
        ref, n = "filter_stable", n + 1
    if "symmetry" in intents:
        steps.append({"step": n, "tool": "analyze_symmetry",
                      "args": {"cif_paths": {"$from": ref}}})
        n += 1
    if "compare" in intents:
        steps.append({"step": n, "tool": "compare_with_experiment",
                      "args": {"cif_paths": {"$from": ref}}})
    return steps


# ---------- 节点实现 ----------

def intent_node(state: AgentState) -> dict:
    """意图识别模块：LLM 分类 + 规则兜底。"""
    user_input = state["user_input"]
    intent, formula = None, None

    if has_api_key():
        try:
            resp = get_llm().invoke(
                INTENT_PROMPT.format(user_input=user_input[:500])
            )
            data = _extract_json(resp.content)
            if isinstance(data, dict):
                intent = data.get("intent")
                formula = data.get("formula")
                if intent not in _VALID_INTENTS:
                    intent = None
        except Exception:
            pass

    if intent is None:
        intents, formula2 = _rule_intent(user_input)
        intent = intents[0]
        formula = formula or formula2
    return {"intent": intent, "formula": formula}


def planner_node(state: AgentState) -> dict:
    """任务规划引擎：LLM 产出 plan，白名单校验，失败走规则兜底。"""
    intent, formula = state["intent"], state.get("formula")
    plan: list | None = None

    if has_api_key() and intent != "explain":
        try:
            formulas = ", ".join(list_formulas())
            resp = get_llm().invoke(PLANNER_PROMPT.format(
                formulas=formulas, intent=intent, formula=formula or "null",
                user_input=state["user_input"][:500],
            ))
            data = _extract_json(resp.content)
            if isinstance(data, list) and data:
                plan = [
                    step for step in data
                    if isinstance(step, dict) and step.get("tool") in TOOL_MAP
                ] or None
        except Exception:
            plan = None

    if plan is None:
        if intent == "explain" and not has_api_key():
            intents = ["explain"]
        else:
            intents, _ = _rule_intent(state["user_input"])  # 组合意图重新扫描
        plan = _rule_plan(intents, formula)

    return {"plan": plan, "cursor": 0, "tool_results": []}


def _resolve_args(args: dict, tool_results: list[dict]) -> dict:
    """解析 {"$from": "<tool>"} 引用 → 前序工具的输出。"""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, dict) and "$from" in v:
            ref = v["$from"]
            prior = next((tr["data"] for tr in reversed(tool_results)
                          if tr["tool"] == ref and tr["ok"]), None)
            resolved[k] = prior if prior is not None else []
        else:
            resolved[k] = v
    return resolved


def executor_node(state: AgentState) -> dict:
    """工具调度 Agent：每步执行 plan[cursor]，cursor+1（条件边决定是否循环）。"""
    plan = state["plan"]
    cursor = state["cursor"]
    results = list(state.get("tool_results") or [])

    if cursor < len(plan):
        step = plan[cursor]
        tool_name = step["tool"]
        args = _resolve_args(step.get("args") or {}, results)
        try:
            data = TOOL_MAP[tool_name].invoke(args)
            results.append({"tool": tool_name, "args": args, "ok": True, "data": data})
        except Exception as e:
            results.append({"tool": tool_name, "args": args, "ok": False,
                            "data": {"error": str(e)}})
        return {"tool_results": results, "cursor": cursor + 1}
    return {"cursor": cursor}


def route_after_executor(state: AgentState) -> str:
    """条件边：还有剩余步骤 → executor；否则 → summarizer。"""
    return "executor" if state["cursor"] < len(state["plan"]) else "summarizer"


def _fallback_summary(state: AgentState) -> str:
    """LLM 不可用时的规则模板汇总（保证表格照样输出）。"""
    lines = [f"**意图**: {state['intent']}  |  **计划步数**: {len(state['plan'])}", ""]
    for tr in state.get("tool_results") or []:
        lines.append(f"### `{tr['tool']}` {'✅' if tr['ok'] else '❌'}")
        data = tr["data"]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys = [k for k in data[0] if k != "cif_path"][:6]
            header = "| " + " | ".join(keys) + " |"
            sep = "|" + "---|" * len(keys)
            body = "\n".join(
                "| " + " | ".join(
                    str(r.get(k, ""))[:24] for k in keys) + " |"
                for r in data[:10]
            )
            lines += [header, sep, body, ""]
        else:
            lines += [f"```json\n{json.dumps(data, ensure_ascii=False, default=str)[:800]}\n```", ""]
    return "\n".join(lines)


def summarizer_node(state: AgentState) -> dict:
    """分析汇总 Agent：tool_results → Markdown 报告。"""
    answer = None
    if has_api_key() and state.get("tool_results"):
        try:
            messages = [
                AIMessage(content=SYSTEM_PROMPT.format(
                    n_structures=len(load_index()))),
                HumanMessage(content=SUMMARIZER_PROMPT.format(
                    user_input=state["user_input"],
                    intent=state["intent"],
                    tool_results=json.dumps(
                        state["tool_results"], ensure_ascii=False, default=str)[:6000],
                )),
            ]
            resp = get_llm().invoke(messages)
            answer = resp.content
        except Exception:
            answer = None

    if not answer:
        answer = _fallback_summary(state)

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "error": None,
    }


# ---------- 图构建 ----------

def build_graph():
    """意图 → 规划 → 执行循环 → 汇总（经典 StateGraph API，跨版本稳定）。"""
    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_after_executor,
                                {"executor": "executor", "summarizer": "summarizer"})
    graph.add_edge("summarizer", END)
    return graph.compile()


def stream_agent(state: AgentState) -> tuple[Generator[dict, None, None], Any]:
    """
    流式执行：返回 (更新生成器, 编译后的图)。

    用法（UI 展示轨迹 + 取最终值）：
        gen, app = stream_agent(initial)
        for event in gen:            # {"node": ..., "update": ...}
            render(event)
        final = app.invoke(initial)  # 或直接用 run_agent 一次性执行
    """
    app = build_graph()

    def _gen():
        for update in app.stream(state):
            for node_name, node_update in update.items():
                yield {"node": node_name, "update": node_update}

    return _gen(), app


def run_agent(user_input: str, history: list | None = None) -> AgentState:
    """一次性执行（smoke test / 简单调用）。"""
    app = build_graph()
    initial: AgentState = {
        "messages": history or [],
        "user_input": user_input,
        "intent": "",
        "formula": None,
        "plan": [],
        "tool_results": [],
        "cursor": 0,
        "final_answer": "",
        "error": None,
    }
    return app.invoke(initial)
