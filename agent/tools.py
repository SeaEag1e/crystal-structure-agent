# -*- coding: utf-8 -*-
"""Agent 工具集：结构库查询 / M3GNet 打分 / 对称分析 / 稳定性筛选 / 实验对比 / GPU 生成。"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from core.scoring import batch_score
from core.structures import (
    REFERENCE_CIF,
    PROJECT_ROOT,
    load_index,
    structure_from_cif,
)
from core.symmetry import analyze_structure


def _resolve(cif_path: str) -> str:
    """把 index.csv 里的相对路径解析为绝对路径。"""
    p = PROJECT_ROOT / cif_path
    return str(p) if p.exists() else cif_path


def _as_cif_path(item: str | dict) -> str:
    """上游工具输出行可能是 dict（含 cif_path 字段）—— 兼容两种输入。"""
    if isinstance(item, dict):
        return str(item.get("cif_path", ""))
    return str(item)


@tool
def query_structure_library(formula: str = "", limit: int = 8) -> list[dict]:
    """Query the pre-generated structure library (DiffCSP++ output + M3GNet cached scores).

    Args:
        formula: chemical formula filter, e.g. "GaPS3I". Empty = all formulas.
        limit: max number of rows to return.

    Returns list of dicts: formula, cif_path, n_atoms, lattice params, volume,
    eform_eV_per_atom (cached), spg_symbol, is_polar, in_top30, rank.
    """
    df = load_index()
    if formula:
        df = df[df["formula"].str.lower() == formula.strip().lower()]
    df = df.sort_values("eform_eV_per_atom").head(max(1, int(limit)))
    cols = ["formula", "cif_path", "n_atoms", "a_A", "b_A", "c_A", "volume_A3",
            "eform_eV_per_atom", "spg_symbol", "spg_number", "is_polar",
            "in_top30", "rank"]
    return df[[c for c in cols if c in df.columns]].to_dict("records")


@tool
def score_structures(cif_paths: list) -> list[dict]:
    """Score structures with the M3GNet ML interatomic potential (formation energy, eV/atom).

    Args:
        cif_paths: list of CIF paths or upstream result rows (dicts with "cif_path").

    Returns list of dicts: cif_path, eform_eV_per_atom, method, is_stable (eform < 0).
    """
    out = []
    for path in cif_paths:
        try:
            s = structure_from_cif(_resolve(_as_cif_path(path)))
            eform, method = batch_score([s])[0]
            out.append({
                "cif_path": path,
                "eform_eV_per_atom": eform,
                "method": method,
                "is_stable": (eform is not None and eform < 0.0),
            })
        except Exception as e:
            out.append({"cif_path": path, "eform_eV_per_atom": None,
                        "method": "failed", "is_stable": None, "error": str(e)})
    return out


@tool
def analyze_symmetry(cif_paths: list, symprec: float = 0.1) -> list[dict]:
    """Analyze space group / point group / polarity via spglib.

    Args:
        cif_paths: list of CIF paths or upstream result rows (dicts with "cif_path").
        symprec: symmetry tolerance in Angstrom (0.1 recommended for generated structures).

    Returns list of dicts: cif_path, spg_number, spg_symbol, point_group, is_polar.
    """
    out = []
    for path in cif_paths:
        try:
            s = structure_from_cif(_resolve(_as_cif_path(path)))
            sym = analyze_structure(s, symprec=symprec)
            out.append({"cif_path": path, **sym})
        except Exception as e:
            out.append({"cif_path": path, "error": str(e)})
    return out


@tool
def filter_stable(candidates: list[dict], threshold: float = 0.0,
                  top_k: int = 10) -> list[dict]:
    """Filter and rank candidate structures by formation energy (stability).

    Keeps entries with eform < threshold, sorts ascending (most stable first),
    keeps at most 2 structures per chemical formula for diversity.

    Args:
        candidates: dicts that contain "eform_eV_per_atom" and "cif_path"
                    (output of score_structures or query_structure_library).
        threshold: stability cutoff in eV/atom (0.0 = negative eform is stable).
        top_k: max results.

    Returns the filtered, ranked candidate dicts (with a "stability_rank" field).
    """
    valid = [c for c in candidates
             if c.get("eform_eV_per_atom") is not None
             and float(c["eform_eV_per_atom"]) < threshold]
    valid.sort(key=lambda c: float(c["eform_eV_per_atom"]))

    per_formula: dict[str, int] = {}
    selected = []
    for c in valid:
        f = c.get("formula", "?")
        if per_formula.get(f, 0) < 2:
            per_formula[f] = per_formula.get(f, 0) + 1
            selected.append({**c, "stability_rank": len(selected) + 1})
        if len(selected) >= top_k:
            break
    return selected


@tool
def compare_with_experiment(cif_paths: list) -> dict:
    """Compare generated structures against the experimental GaPSe3I reference (space group C2/c).

    Args:
        cif_paths: list of CIF paths or upstream result rows (dicts with "cif_path").

    Returns dict with the experimental reference info and per-structure
    lattice-parameter deviation (%).
    """
    if not REFERENCE_CIF.exists():
        return {"error": "experimental reference CIF not found"}

    ref = structure_from_cif(str(REFERENCE_CIF))
    ref_eform, ref_method = batch_score([ref])[0]
    result: dict[str, Any] = {
        "reference": {
            "cif_path": str(REFERENCE_CIF),
            "formula": ref.composition.reduced_formula,
            "n_atoms": len(ref),
            "a_A": round(ref.lattice.a, 3),
            "b_A": round(ref.lattice.b, 3),
            "c_A": round(ref.lattice.c, 3),
            "volume_A3": round(ref.lattice.volume, 2),
            "eform_eV_per_atom": ref_eform,
            "eform_method": ref_method,
            "space_group": "C2/c (15), centrosymmetric, non-polar",
        },
        "comparisons": [],
    }
    for path in cif_paths:
        try:
            s = structure_from_cif(_resolve(_as_cif_path(path)))
            result["comparisons"].append({
                "cif_path": path,
                "formula": s.composition.reduced_formula,
                "n_atoms": len(s),
                "a_dev_pct": round(100 * (s.lattice.a - ref.lattice.a) / ref.lattice.a, 2),
                "b_dev_pct": round(100 * (s.lattice.b - ref.lattice.b) / ref.lattice.b, 2),
                "c_dev_pct": round(100 * (s.lattice.c - ref.lattice.c) / ref.lattice.c, 2),
                "volume_dev_pct": round(
                    100 * (s.lattice.volume - ref.lattice.volume) / ref.lattice.volume, 2),
            })
        except Exception as e:
            result["comparisons"].append({"cif_path": path, "error": str(e)})
    return result


@tool
def generate_local_gpu(formula: str, num_structures: int = 3) -> dict:
    """Generate NEW crystal structures with the DiffCSP++ diffusion model.

    Requires a local GPU environment with DiffCSP++ weights (DIFFCSP_ROOT env var).
    On CPU-only servers (e.g. Hugging Face Spaces) this returns an availability
    notice instead of generating - the agent degrades gracefully.

    Args:
        formula: target composition, e.g. "GaPS3I".
        num_structures: number of candidate structures to sample.

    Returns dict with "available" flag and either generated structures or a notice.
    """
    from core.generation import is_diffcsp_ready, is_gpu_available

    if not is_gpu_available() or not is_diffcsp_ready():
        return {
            "available": False,
            "message": (
                "DiffCSP++ generation requires a local GPU environment. "
                "This demo server is CPU-only, so the pretrained library "
                "(186 structures, 16 formulas) is served instead. "
                "See README 'Local GPU mode' to run generation yourself."
            ),
        }

    try:
        from core.generation import generate_structures

        pairs = generate_structures(formula, num_structures=num_structures,
                                    output_dir=str(PROJECT_ROOT / "data" / "generated"))
        generated = []
        for s, info in pairs:
            eform, method = batch_score([s])[0]
            sym = analyze_structure(s)
            generated.append({
                "formula": info["formula"],
                "cif_path": info.get("cif_path", ""),
                "a_A": round(s.lattice.a, 3),
                "b_A": round(s.lattice.b, 3),
                "c_A": round(s.lattice.c, 3),
                "eform_eV_per_atom": eform,
                "spg_symbol": sym["spg_symbol"],
                "is_polar": sym["is_polar"],
            })
        return {"available": True, "structures": generated}
    except Exception as e:
        return {"available": False, "message": f"generation failed: {e}"}


TOOL_MAP: dict[str, Any] = {
    "query_structure_library": query_structure_library,
    "score_structures": score_structures,
    "analyze_symmetry": analyze_symmetry,
    "filter_stable": filter_stable,
    "compare_with_experiment": compare_with_experiment,
    "generate_local_gpu": generate_local_gpu,
}
