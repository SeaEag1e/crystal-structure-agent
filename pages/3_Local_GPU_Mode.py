# -*- coding: utf-8 -*-
"""本地 GPU 生成模式页：有 CUDA + DiffCSP++ 权重时跑完整扩散生成，否则显示指引。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from core.generation import is_diffcsp_ready, is_gpu_available
from core.scoring import batch_score
from core.symmetry import analyze_structure

st.set_page_config(page_title="Local GPU Mode", page_icon="⚙️", layout="wide")
st.title("⚙️ Local GPU Mode — DiffCSP++ Generation")

gpu_ok = is_gpu_available()
weights_ok = is_diffcsp_ready()

if gpu_ok and weights_ok:
    st.success("CUDA GPU + DiffCSP++ weights detected — full generation enabled.")
elif not gpu_ok:
    st.warning(
        "**CPU-only environment.** The diffusion sampler needs a GPU (1000 denoising "
        "steps per structure). This hosted demo serves the **pre-generated library** "
        "instead — see the Structure Library page."
    )
else:
    st.warning(
        "**DiffCSP++ weights not found.** Set `DIFFCSP_ROOT` to your DiffCSP-main "
        "checkout containing `pretrained/mp_csp`."
    )

if gpu_ok and weights_ok:
    with st.form("gen_form"):
        c1, c2 = st.columns([2, 1])
        formula = c1.text_input("Target formula", value="GaPS3I")
        num = c2.number_input("Structures", min_value=1, max_value=8, value=3)
        submitted = st.form_submit_button("🚀 Generate & score", use_container_width=True)

    if submitted:
        from core.generation import generate_structures

        with st.status("DiffCSP++ sampling (1000 denoising steps)...", expanded=True) as box:
            pairs = generate_structures(
                formula, num_structures=int(num),
                output_dir=str(Path(__file__).resolve().parent.parent / "data" / "generated"),
            )
            box.update(label="Sampling done", state="complete", expanded=False)

        for i, (s, info) in enumerate(pairs, 1):
            (eform, method), sym = batch_score([s])[0], analyze_structure(s)
            st.markdown(
                f"**#{i}** `{info['formula']}` — a={s.lattice.a:.3f} b={s.lattice.b:.3f} "
                f"c={s.lattice.c:.3f} Å · eform **{eform:+.4f} eV/atom** (`{method}`) · "
                f"space group {sym['spg_symbol']} · polar: {'Yes' if sym['is_polar'] else 'No'}"
            )
else:
    st.markdown("### How the offline pipeline works")
    st.markdown(
        """
```mermaid
flowchart LR
    F[Formula<br/>e.g. GaPS3I] --> D[DiffCSP++<br/>mp_csp checkpoint]
    D -->|1000-step reverse diffusion<br/>lattice + frac coords| R[Candidate structures]
    R --> M[M3GNet<br/>formation energy eV/atom]
    M --> SP[spglib<br/>space group / polarity]
    SP --> T[(TOP-30 library<br/>bundled in this app)]
```
"""
    )
    st.markdown("### Run it locally")
    st.code(
        """
# 1. Clone DiffCSP++ and place the mp_csp checkpoint
git clone https://github.com/jiao-rui/DiffCSP-plusplus  # (or your checkout)
export DIFFCSP_ROOT=/path/to/DiffCSP-main

# 2. Run the CLI generator
python scripts/generate_gpu.py --formula GaPS3I -n 4

# 3. Or start this app on a GPU machine
streamlit run app.py
""",
        language="bash",
    )
