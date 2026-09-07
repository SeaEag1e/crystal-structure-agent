# -*- coding: utf-8 -*-
"""Crystal Structure Agent — Streamlit 首页：项目简介 + 架构图 + 环境自检。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from core.scoring import model_status

st.set_page_config(page_title="Crystal Structure Agent", page_icon="💎",
                   layout="wide")

st.title("💎 Crystal Structure Agent")
st.markdown(
    """
    AI-driven crystal structure prediction & screening for MgPS₃-type halide
    materials — **DiffCSP++ diffusion model** generates candidate structures,
    **M3GNet** ML interatomic potential scores their stability, and a
    **LangGraph multi-agent system** (intent recognition → task planning →
    tool orchestration → analysis) coordinates everything through natural
    language.
    """
)

# ---- 架构图 ----
st.markdown("### Architecture")
st.markdown(
    """
```mermaid
flowchart LR
    U[User query] --> IR[Intent Recognition<br/>LLM + rule fallback]
    IR --> P[Task Planner<br/>LLM structured plan]
    P --> E[Tool Executor<br/>LangGraph loop]
    E -->|1| T1[query_structure_library]
    E -->|2| T2[score_structures<br/>M3GNet]
    E -->|3| T3[analyze_symmetry<br/>spglib]
    E -->|4| T4[filter_stable]
    E -->|5| T5[compare_with_experiment]
    E -->|6| T6[generate_local_gpu<br/>DiffCSP++ GPU]
    E --> A[Analyst<br/>Markdown report]
    A --> O[Answer]

    subgraph Offline [Offline pipeline · local GPU]
        G[DiffCSP++ 1000-step<br/>denoising sampling] --> S[M3GNet scoring<br/>+ spglib symmetry] --> L[(Structure library<br/>186 CIF / 16 formulas)]
    end
    L -.-> T1
```
"""
)

# ---- 环境自检 ----
st.markdown("### Environment self-check")
c1, c2, c3 = st.columns(3)

with c1:
    status = model_status()
    if status["loaded"]:
        st.success(f"**M3GNet model** loaded\n\n`{status['name']}` ({status['source']})")
    else:
        st.warning("**M3GNet not available**\n\nWill fall back to heuristic scoring")

with c2:
    import os
    if os.environ.get("ZHIPU_API_KEY"):
        st.success("**GLM-4-Flash API** configured\n\nAgent LLM reasoning enabled")
    else:
        st.info("**GLM-4-Flash key not set**\n\nAgent runs in rule-based fallback mode")

with c3:
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    if gpu:
        st.success("**CUDA GPU** available\n\nDiffCSP++ generation enabled")
    else:
        st.info("**CPU-only** server\n\nGeneration page shows library mode")

# ---- 页面导航说明 ----
st.markdown("### Explore")
n1, n2, n3 = st.columns(3)
with n1:
    st.page_link("pages/1_Agent_Chat.py", label="🤖 **Agent Chat**",
                 icon="🤖")
    st.caption("Natural-language analysis: *\"find the most stable structures and "
               "check their polarity\"*")
with n2:
    st.page_link("pages/2_Structure_Library.py", label="🔬 **Structure Library**",
                 icon="🔬")
    st.caption("Browse 186 candidate structures with cached M3GNet scores, "
               "3D rendering and symmetry analysis")
with n3:
    st.page_link("pages/3_Local_GPU_Mode.py", label="⚙️ **Local GPU Mode**",
                 icon="⚙️")
    st.caption("Run the full DiffCSP++ diffusion generation on your own GPU")

st.divider()
st.caption(
    "Models: DiffCSP++ (Jiao et al., NeurIPS 2023) · M3GNet (Chen & Ong, Nat. Comput. Sci. 2022) "
    "· LLM: GLM-4-Flash (Zhipu AI). Built with LangGraph + Streamlit."
)
