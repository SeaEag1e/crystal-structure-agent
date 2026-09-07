# -*- coding: utf-8 -*-
"""形成能打分：M3GNet 图神经网络推理 + 元素焓值 fallback（三级降级）。"""
from __future__ import annotations

import warnings
from pathlib import Path

from pymatgen.core import Structure

from core.structures import DATA_DIR

warnings.filterwarnings("ignore")

# 内嵌权重目录（构建时打包进仓库，避免运行时从 HF Hub 下载）
LOCAL_WEIGHT_DIR = DATA_DIR / "models" / "matgl"
# 降级链：M3GNet（显式三体相互作用）→ MEGNet（二体）→ 启发式焓值
MODEL_CANDIDATES = ["M3GNet-Eform-MP-2019.4.1", "MEGNet-MP-2018.6.1-Eform"]

_model = None
_model_name: str | None = None


def get_m3gnet_model():
    """全局单例加载形成能模型。返回 (model|None, model_name|None)。"""
    global _model, _model_name
    if _model is not None:
        return _model, _model_name

    try:
        import matgl
        import torch

        torch.set_num_threads(2)  # HF Spaces 2 vCPU 防争抢
    except ImportError:
        return None, None

    # 1) 内嵌本地权重（离线优先，冷启动快）
    if LOCAL_WEIGHT_DIR.exists():
        for cand in MODEL_CANDIDATES:
            local = LOCAL_WEIGHT_DIR / cand
            if local.is_dir():
                try:
                    _model = matgl.load_model(str(local))
                    _model_name = cand
                    return _model, _model_name
                except Exception:
                    continue

    # 2) HF Hub 在线加载（本地开发环境）
    for cand in MODEL_CANDIDATES:
        try:
            _model = matgl.load_model(cand)
            _model_name = cand
            return _model, _model_name
        except Exception:
            continue

    return None, None


def model_status() -> dict:
    """环境自检信息（首页展示用）。"""
    model, name = get_m3gnet_model()
    return {
        "loaded": model is not None,
        "name": name,
        "source": "bundled" if (DATA_DIR / "models" / "matgl" / (name or "_") ).exists() else "hub",
    }


# ---- fallback：元素原子化焓（kJ/mol），仅当全部 ML 模型不可用时做粗略排序 ----
ELEMENT_FORMATION_ENTHALPY = {
    'H': 218.0, 'Li': 159.3, 'Be': 324.0, 'B': 562.7, 'C': 716.7,
    'N': 472.7, 'O': 249.2, 'F': 79.0, 'Na': 107.5, 'Mg': 147.8,
    'Al': 326.4, 'Si': 450.0, 'P': 314.6, 'S': 277.0, 'Cl': 121.7,
    'K': 89.2, 'Ca': 178.2, 'Ti': 468.6, 'Cr': 416.6, 'Mn': 423.0,
    'Fe': 416.3, 'Co': 424.7, 'Ni': 430.1, 'Cu': 337.4, 'Zn': 130.4,
    'Ga': 277.0, 'Ge': 376.6, 'As': 297.7, 'Se': 227.0, 'Br': 111.9,
    'Rb': 80.9, 'Sr': 164.4, 'Zr': 503.4, 'Mo': 658.1, 'Ru': 649.8,
    'Rh': 550.0, 'Pd': 393.3, 'Ag': 284.9, 'Cd': 112.0,
    'In': 246.0, 'Sn': 301.2, 'Sb': 262.3, 'Te': 196.0, 'I': 106.8,
    'Cs': 76.1, 'Ba': 180.0, 'La': 431.0, 'Ce': 464.0,
    'Yb': 159.0, 'Hf': 715.0, 'Ta': 782.0, 'W': 848.0, 'Pt': 565.0,
    'Au': 366.0, 'Hg': 61.4, 'Tl': 182.8, 'Pb': 195.2, 'Bi': 207.1,
}


def _heuristic_eform(structure: Structure) -> float:
    """元素原子化焓求和的粗略估算（kJ/mol → eV/atom）。"""
    total = sum(
        ELEMENT_FORMATION_ENTHALPY.get(str(site.specie), 250.0)
        for site in structure
    )
    return round(total / len(structure) / 96.485, 5)


def predict_formation_energy(structure: Structure) -> tuple[float | None, str]:
    """
    形成能预测。返回 (eform_eV_per_atom | None, method)。

    method ∈ {"M3GNet-Eform-MP-2019.4.1", "MEGNet-MP-2018.6.1-Eform",
              "heuristic-enthalpy", "failed"}
    """
    model, name = get_m3gnet_model()
    if model is not None:
        try:
            import torch
            with torch.inference_mode():
                return round(float(model.predict_structure(structure)), 6), name
        except Exception:
            pass
    try:
        return _heuristic_eform(structure), "heuristic-enthalpy"
    except Exception:
        return None, "failed"


def batch_score(structures: list[Structure]) -> list[tuple[float | None, str]]:
    """批量打分（逐条推理 + try/except 隔离，单条失败不影响整体）。"""
    return [predict_formation_energy(s) for s in structures]


def score_cif(path: str | Path) -> tuple[float | None, str]:
    """便捷入口：直接对 CIF 文件打分。"""
    return predict_formation_energy(Structure.from_file(str(path)))
