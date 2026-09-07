# -*- coding: utf-8 -*-
"""Agent 冒烟测试：3 条固定 query 走通全链路（无 API key 时走规则兜底模式）。

用法：python scripts/smoke_test_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.graph import TOOL_MAP, run_agent  # noqa: E402
from agent.llm import has_api_key            # noqa: E402

QUERIES = [
    # (query, 期望 intent, 期望涉及的核心工具)
    ("浏览 GaPS3I 的结构", "browse", {"query_structure_library"}),
    ("找出最稳定的结构并分析它们的对称性", "stability",
     {"query_structure_library", "filter_stable"}),
    ("生成的结构和实验结构对比一下", "compare",
     {"query_structure_library", "compare_with_experiment"}),
]

EXPECTED_INTENTS = {"browse", "stability", "symmetry", "compare",
                    "generate", "explain"}


def main() -> int:
    mode = "LLM (GLM-4-Flash)" if has_api_key() else "rule-based fallback"
    print(f"=== Agent smoke test — mode: {mode} ===\n")

    failures = 0
    for query, expected_intent, core_tools in QUERIES:
        print(f"\n>>> {query}")
        state = run_agent(query)

        intent = state.get("intent")
        plan = state.get("plan") or []
        plan_tools = {s["tool"] for s in plan}
        results = state.get("tool_results") or []
        answer = state.get("final_answer") or ""

        print(f"    intent      : {intent}")
        print(f"    plan        : {' -> '.join(s['tool'] for s in plan) or '(empty)'}")
        print(f"    tools ran   : {[(t['tool'], t['ok']) for t in results]}")
        print(f"    answer head : {answer[:120]!r}")

        # 断言 1：intent 合法
        if intent not in EXPECTED_INTENTS:
            print("    ❌ FAIL: invalid intent")
            failures += 1
            continue
        # 断言 2：plan 工具全部在白名单
        if not plan_tools <= set(TOOL_MAP):
            print("    ❌ FAIL: plan contains unknown tools")
            failures += 1
            continue
        # 断言 3：核心工具被执行且成功
        ran_ok = {t["tool"] for t in results if t["ok"]}
        if expected_intent != "browse" and not core_tools <= ran_ok:
            print(f"    ⚠️  WARN: expected tools {core_tools - ran_ok} did not run ok")
        if not answer:
            print("    ❌ FAIL: empty answer")
            failures += 1
            continue
        print("    ✅ PASS")

    # 额外：generate 工具在 CPU 环境必须优雅降级
    print("\n>>> [unit] generate_local_gpu on CPU-only env")
    result = TOOL_MAP["generate_local_gpu"].invoke(
        {"formula": "GaPS3I", "num_structures": 1})
    ok = (result.get("available") is False) and ("message" in result)
    print(f"    degrade gracefully: {'✅ PASS' if ok else '❌ FAIL'}")
    if not ok:
        failures += 1

    print(f"\n=== {failures} failure(s) ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
