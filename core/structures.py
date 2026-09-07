# -*- coding: utf-8 -*-
"""结构库 IO：index.csv 读取、CIF 解析、P1 CIF 文本生成。"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from pymatgen.core import Structure

# 仓库根目录（core/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_CSV = DATA_DIR / "index.csv"
REFERENCE_CIF = DATA_DIR / "reference" / "GaPSe3I_experimental.cif"


def load_index() -> pd.DataFrame:
    """读取结构库索引（预计算 eform/空间群/极性等元数据）。"""
    return pd.read_csv(INDEX_CSV)


def list_formulas() -> list[str]:
    """库中全部化学式（按字母序）。"""
    df = load_index()
    return sorted(df["formula"].unique().tolist())


def structure_from_cif(path: str | Path) -> Structure:
    """从 CIF 文件解析 pymatgen Structure。"""
    return Structure.from_file(str(path))


def cif_to_text(struct: Structure) -> str:
    """
    生成 P1 CIF 文本（不做对称性约化，所有原子显式写出）。

    强制 P1 的原因：DiffCSP++ 生成的结构原子位置带数值噪声，
    spglib 对称约化在不同软件/容差下结果不稳定；显式坐标保证
    任何下游程序解析到完全相同的原子列表（可复现性优先）。
    """
    a, b, c = struct.lattice.a, struct.lattice.b, struct.lattice.c
    alpha, beta, gamma = struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma
    lines = [
        f"data_{struct.composition.reduced_formula}\n",
        "_audit_creation_method 'crystal-structure-agent'\n",
        f"_chemical_formula_sum '{struct.composition.reduced_formula}'\n",
        f"_cell_length_a {a:.6f}\n",
        f"_cell_length_b {b:.6f}\n",
        f"_cell_length_c {c:.6f}\n",
        f"_cell_angle_alpha {alpha:.6f}\n",
        f"_cell_angle_beta {beta:.6f}\n",
        f"_cell_angle_gamma {gamma:.6f}\n",
        f"_cell_volume {struct.lattice.volume:.6f}\n",
        "_space_group_name_H-M_alt 'P 1'\n",
        "_symmetry_Int_Tables_number 1\n",
        "loop_\n  _symmetry_equiv_pos_site_id\n  _symmetry_equiv_pos_as_xyz\n  1 'x, y, z'\n",
        "loop_\n  _atom_site_label\n  _atom_site_type_symbol\n"
        "  _atom_site_fract_x\n  _atom_site_fract_y\n  _atom_site_fract_z\n"
        "  _atom_site_occupancy\n",
    ]
    for i, site in enumerate(struct):
        species = str(site.specie)
        fx, fy, fz = site.frac_coords
        lines.append(
            f"  {species}{i + 1} {species} {fx:10.6f} {fy:10.6f} {fz:10.6f} 1.0\n"
        )
    return "".join(lines)


def load_reference_structure() -> Structure | None:
    """加载实验参照结构（GaPSe3I, C2/c）。"""
    if REFERENCE_CIF.exists():
        return Structure.from_file(str(REFERENCE_CIF))
    return None
