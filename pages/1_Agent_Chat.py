# -*- coding: utf-8 -*-
"""Agent 对话页：自然语言 → LangGraph 多 Agent 轨迹 → Markdown 报告。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agent.graph import build_graph
from agent.llm import has_api_key
from agent.state import AgentState

st.set_page_config(page_title="Agent Chat", page_icon="🤖", layout="wide")
st.title("🤖 Agent Chat")

if not has_api_key():
    st.info(
        "**Fallback mode** — `ZHIPU_API_KEY` not configured, so the agent runs with "
        "rule-based intent recognition / planning and template summaries. "
        "Tool execution (M3GNet scoring, symmetry analysis) still works fully."
    )

if "history" not in st.session_state:
    st.session_state.history = []          # [(user, answer)]
if "trace" not in st.session_state:
    st.session_state.trace = []            # 最后一次运行的 (node, update)

# ---- 侧栏示例 ----
with st.sidebar:
    st.markdown("### Example queries")
    examples = [
        "浏览 GaPS3I 的结构",
        "找出最稳定的结构并分析它们的对称性",
        "List the top-5 most stable structures across all formulas",
        "生成的结构和实验结构（C2/c）对比一下",
        "Show me polar structures of AlPS3I",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_input = ex
            st.rerun()

# ---- 渲染历史 ----
for user_msg, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(answer)

# ---- 输入 ----
user_input = st.session_state.pop("pending_input", None) or st.chat_input(
    "Ask about structures, stability, symmetry..."
)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        initial: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "formula": None,
            "plan": [],
            "tool_results": [],
            "cursor": 0,
            "final_answer": "",
            "error": None,
        }
        app = build_graph()

        with st.status("Agent running...", expanded=True) as status_box:
            final_state = app.invoke(initial)
            status_box.update(label="Agent finished", state="complete",
                              expanded=False)

        # 从最终状态渲染轨迹（invoke 一次拿全量，避免重复执行）
        with st.status("Agent trace", expanded=False) as trace_box:
            st.markdown(
                f"🧭 **Intent**: `{final_state.get('intent')}`"
                + (f" · formula: `{final_state.get('formula')}`"
                   if final_state.get("formula") else "")
            )
            plan = final_state.get("plan") or []
            step_desc = " → ".join(f"`{s['tool']}`" for s in plan) or "*(explain-only)*"
            st.markdown(f"📋 **Plan** ({len(plan)} steps): {step_desc}")
            for tr in final_state.get("tool_results") or []:
                icon = "✅" if tr.get("ok") else "❌"
                st.markdown(f"🔧 **Executed** `{tr.get('tool')}` {icon}")
            trace_box.update(label="Agent trace (complete)", expanded=False)

        answer = final_state.get("final_answer") or "No answer produced."
        st.markdown(answer)

    st.session_state.history.append((user_input, answer))
    st.session_state.trace = final_state.get("tool_results") or []
