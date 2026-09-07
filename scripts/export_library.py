# -*- coding: utf-8 -*-
"""
一次性数据管线：把本地 DiffCSP++ 生成结果导出为仓库自带的静态结构库。

输入（jiegouyuce 项目目录，按需修改 SOURCE_*）：
  1. batch_diffcsp_outputs/{FORMULA}/*.cif        — 原始生成 CIF（排除 TOP30_BEST）
  2. diffcsp_final_results_m3gnet/ORIGINAL_6atoms_TOP30/*.cif — TOP30（带 rank 前缀）
  3. diffcsp_final_results_m3gnet/summary_TOP30_with_symmetry.csv — rank 对照表

输出（仓库内）：
  data/structures/{FORMULA}/*.cif
  data/index.csv        — 知识库底座：formula/cif_path/晶格/eform/spg/is_polar/in_top30/rank
  data/top30_summary.csv
"""
from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.scoring import predict_formation_energy  # noqa: E402
from core.symmetry import analyze_structure         # noqa: E402

# ---- 源目录（本机路径，仅导出时使用） ----
SOURCE_BATCH = Path(r"G:\TRAE\jiegouyuce\batch_diffcsp_outputs")
SOURCE_TOP30_DIR = Path(r"G:\TRAE\jiegouyuce\diffcsp_final_results_m3gnet\ORIGINAL_6atoms_TOP30")
SOURCE_TOP30_CSV = Path(r"G:\TRAE\jiegouyuce\diffcsp_final_results_m3gnet\summary_TOP30_with_symmetry.csv")

# ---- 目标 ----
OUT_STRUCT_DIR = REPO_ROOT / "data" / "structures"
OUT_INDEX = REPO_ROOT / "data" / "index.csv"
OUT_TOP30_CSV = REPO_ROOT / "data" / "top30_summary.csv"


def collect_sources() -> list[tuple[Path, str]]:
    """返回 [(cif_path, 化学式目录名), ...]；TOP30 文件名形如 rank01_... 保留原文件名。"""
    items: list[tuple[Path, str]] = []
    seen_names: set[tuple[str, str]] = set()

    # TOP30 优先（文件名带 rank，不冲突）
    if SOURCE_TOP30_DIR.is_dir():
        for cif in sorted(SOURCE_TOP30_DIR.glob("*.cif")):
            formula_dir = cif.stem.split("_")[1] if "_" in cif.stem else "MISC"
            items.append((cif, formula_dir))
            seen_names.add((formula_dir, cif.name))

    # 原始批量生成
    if SOURCE_BATCH.is_dir():
        for cif in sorted(SOURCE_BATCH.glob("*/*.cif")):
            if "TOP30_BEST" in str(cif):
                continue
            formula_dir = cif.parent.name
            name = cif.name
            if (formula_dir, name) in seen_names:
                name = f"gen_{name}"
            items.append((cif, formula_dir))
            seen_names.add((formula_dir, name))
    return items


def main() -> None:
    sources = collect_sources()
    print(f"发现 {len(sources)} 个源 CIF")

    # rank 对照表：original_cif 文件名 → rank
    rank_map: dict[str, int] = {}
    if SOURCE_TOP30_CSV.exists():
        df_top = pd.read_csv(SOURCE_TOP30_CSV)
        rank_map = {
            str(row["original_cif"]): int(row["rank"])
            for _, row in df_top.iterrows()
        }
        df_top.to_csv(OUT_TOP30_CSV, index=False)

    rows = []
    errors = []
    t0 = time.time()
    for idx, (src, formula_dir) in enumerate(sources, 1):
        try:
            s = Structure.from_file(str(src))
            formula = s.composition.reduced_formula

            # 目标路径：data/structures/{化学式目录}/
            dst_dir = OUT_STRUCT_DIR / formula_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)

            eform, method = predict_formation_energy(s)
            sym = analyze_structure(s)

            rows.append({
                "formula": formula,
                "cif_path": f"data/structures/{formula_dir}/{src.name}",
                "n_atoms": len(s),
                "a_A": round(s.lattice.a, 4),
                "b_A": round(s.lattice.b, 4),
                "c_A": round(s.lattice.c, 4),
                "alpha_deg": round(s.lattice.alpha, 4),
                "beta_deg": round(s.lattice.beta, 4),
                "gamma_deg": round(s.lattice.gamma, 4),
                "volume_A3": round(s.lattice.volume, 3),
                "eform_eV_per_atom": eform,
                "eform_method": method,
                "spg_number": sym["spg_number"],
                "spg_symbol": sym["spg_symbol"],
                "point_group": sym["point_group"],
                "is_polar": sym["is_polar"],
                "in_top30": src.name in rank_map,
                "rank": rank_map.get(src.name, ""),
            })
        except Exception as ex:
            errors.append((str(src), str(ex)))

        if idx % 20 == 0 or idx == len(sources):
            print(f"  [{idx}/{len(sources)}] {time.time() - t0:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values("eform_eV_per_atom").reset_index(drop=True)
    df.to_csv(OUT_INDEX, index=False)

    print(f"\n完成: {len(rows)} 成功 / {len(errors)} 失败 / {time.time() - t0:.0f}s")
    print(f"  index.csv → {OUT_INDEX}")
    print(f"  化学式数: {df['formula'].nunique()} | TOP30 标记: {df['in_top30'].sum()}")
    print(f"  eform 区间: [{df['eform_eV_per_atom'].min():.3f}, "
          f"{df['eform_eV_per_atom'].max():.3f}] eV/atom")
    if errors:
        print("\n失败清单:")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")

    # 校验锚点：rank1 应为 AlPS3I，eform ≈ -0.890
    r1 = df[df["rank"] == 1]
    if not r1.empty:
        row = r1.iloc[0]
        print(f"\n校验 rank1: {row['formula']} eform={row['eform_eV_per_atom']:.6f} "
              f"(期望 AlPS3I ≈ -0.890)")


if __name__ == "__main__":
    main()
