# -*- coding: utf-8 -*-
"""对称性分析：spglib 空间群识别 + 极性判据（铁电性必要条件）。"""
from __future__ import annotations

from pymatgen.core import Structure

# 10 个极性点群：存在唯一极轴 → 满足铁电性必要条件（非中心对称的子集）
POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}

_DEFAULT_SYMPREC = 0.1


def analyze_structure(structure: Structure, symprec: float = _DEFAULT_SYMPREC) -> dict:
    """
    返回 {spg_number, spg_symbol, point_group, is_polar, is_centrosymmetric_hint}。

    注意：DiffCSP++ 原始输出带数值噪声，symprec 取 0.1 宽容识别；
    结果应视为"对称性倾向"而非精确空间群（DFT 弛豫后才收敛）。
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
    point_group = analyzer.get_point_group_symbol()
    return {
        "spg_number": int(analyzer.get_space_group_number()),
        "spg_symbol": analyzer.get_space_group_symbol(),
        "point_group": point_group,
        "is_polar": point_group in POLAR_POINT_GROUPS,
        "symprec": symprec,
    }


def analyze_cif(path: str, symprec: float = _DEFAULT_SYMPREC) -> dict:
    """便捷入口：直接分析 CIF 文件。"""
    return analyze_structure(Structure.from_file(path), symprec=symprec)
