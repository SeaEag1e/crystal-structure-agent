# -*- coding: utf-8 -*-
"""本地 GPU CLI：python scripts/generate_gpu.py --formula GaPS3I -n 4

要求：CUDA GPU + DIFFCSP_ROOT 指向 DiffCSP-main 代码目录（含 pretrained/mp_csp）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.generation import generate_structures, is_diffcsp_ready, is_gpu_available
from core.scoring import batch_score
from core.symmetry import analyze_structure


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffCSP++ GPU generation CLI")
    parser.add_argument("--formula", default="GaPS3I", help="target composition")
    parser.add_argument("-n", "--num", type=int, default=4, help="structures to sample")
    parser.add_argument("--step-lr", type=float, default=1e-5, help="diffusion step size")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "generated"),
                        help="output directory")
    args = parser.parse_args()

    if not is_gpu_available():
        sys.exit("ERROR: CUDA GPU not available - generation requires a GPU.")
    if not is_diffcsp_ready():
        sys.exit("ERROR: DiffCSP++ weights not found. "
                 "Set DIFFCSP_ROOT to the DiffCSP-main directory.")

    pairs = generate_structures(args.formula, num_structures=args.num,
                                step_lr=args.step_lr, output_dir=args.out)
    print(f"\n{'#':<4}{'formula':<12}{'a(A)':<9}{'b(A)':<9}{'c(A)':<9}"
          f"{'eform(eV/atom)':<16}{'spg':<8}{'polar'}")
    for i, (s, info) in enumerate(pairs, 1):
        (eform, _method), sym = batch_score([s])[0], analyze_structure(s)
        print(f"{i:<4}{info['formula']:<12}{s.lattice.a:<9.3f}{s.lattice.b:<9.3f}"
              f"{s.lattice.c:<9.3f}{eform:+.5f}{'':<9}{sym['spg_symbol']:<8}"
              f"{'yes' if sym['is_polar'] else 'no'}")


if __name__ == "__main__":
    main()
