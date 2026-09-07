# -*- coding: utf-8 -*-
"""DiffCSP++ 扩散模型封装（仅本地 GPU 环境可用，在线 Spaces 上优雅降级）。

设计要点：
- torch_geometric / chemparse 均为函数内延迟导入 → Spaces 无 PyG 也能安全 import 本模块
- DIFFCSP_ROOT 环境变量指定 DiffCSP++ 代码与权重根目录，摆脱硬编码路径
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CHEMICAL_SYMBOLS = [
    'X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
    'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
]

DIFFCSP_ROOT = Path(os.environ.get("DIFFCSP_ROOT", "../DiffCSP/DiffCSP-main")).resolve()

_model_cache: dict = {}


def is_gpu_available() -> bool:
    """CUDA 是否可用（决定 Agent 生成工具是否降级）。"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_diffcsp_ready() -> bool:
    """DiffCSP++ 代码与权重是否就位。"""
    return (DIFFCSP_ROOT / "pretrained" / "mp_csp").is_dir()


def _bootstrap_diffcsp_path() -> None:
    os.environ["PROJECT_ROOT"] = str(DIFFCSP_ROOT)
    for sub in ("", "scripts"):
        p = str(DIFFCSP_ROOT / sub) if sub else str(DIFFCSP_ROOT)
        if p not in sys.path:
            sys.path.insert(0, p)


def get_diffcsp_model():
    """加载 mp_csp 生成模型（进程内缓存）。"""
    if "csp" in _model_cache:
        return _model_cache["csp"]

    import importlib
    _bootstrap_diffcsp_path()
    eval_utils = importlib.import_module("eval_utils")

    t0 = time.time()
    model, _, _cfg = eval_utils.load_model(
        DIFFCSP_ROOT / "pretrained" / "mp_csp", load_data=False
    )
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
    _model_cache["csp"] = model
    print(f"[DiffCSP++] model loaded in {time.time() - t0:.1f}s")
    return model


def generate_structures(formula: str, num_structures: int = 3,
                        step_lr: float = 1e-5, output_dir: str | Path | None = None):
    """
    扩散采样生成晶体结构。返回 list[(Structure, info_dict)]。

    1000 步反向扩散：随机晶格(高斯) + 随机坐标(均匀) → CSPNet 逐步去噪。
    """
    import importlib

    import torch
    from torch_geometric.data import Data, Batch
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    from pymatgen.io.cif import CifWriter

    import chemparse

    _bootstrap_diffcsp_path()
    eval_utils = importlib.import_module("eval_utils")

    model = get_diffcsp_model()
    model.eval()

    composition = chemparse.parse_formula(formula)
    chem_list: list[int] = []
    for elem, cnt in composition.items():
        chem_list.extend([CHEMICAL_SYMBOLS.index(elem)] * int(cnt))

    data_list = [
        Data(atom_types=torch.LongTensor(chem_list),
             num_atoms=len(chem_list), num_nodes=len(chem_list))
        for _ in range(num_structures)
    ]
    batch = Batch.from_data_list(data_list)
    if torch.cuda.is_available():
        batch = batch.cuda()

    t0 = time.time()
    outputs, _traj = model.sample(batch, step_lr=step_lr)
    print(f"[DiffCSP++] sampled in {time.time() - t0:.1f}s")

    frac_coords = outputs["frac_coords"].detach().cpu()
    atom_types_out = outputs["atom_types"].detach().cpu()
    num_atoms_tensor = outputs["num_atoms"].detach().cpu()
    lattices = outputs["lattices"].detach().cpu()
    lengths, angles = eval_utils.lattices_to_params_shape(lattices)

    if len(atom_types_out.shape) == 2:  # 概率分布 → argmax
        atom_indices = torch.argmax(atom_types_out, dim=1)
    else:
        atom_indices = atom_types_out.long()

    if output_dir is not None:
        formula_dir = Path(output_dir) / formula
        formula_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start = 0
    for i, n in enumerate(num_atoms_tensor.tolist()):
        n = int(n)
        cur_frac = frac_coords[start:start + n].numpy()
        cur_atom = atom_indices[start:start + n]
        cur_len, cur_ang = lengths[i], angles[i]
        start += n
        try:
            species = [CHEMICAL_SYMBOLS[int(z)] for z in cur_atom.tolist()]
            lat = Lattice.from_parameters(*(cur_len.tolist() + cur_ang.tolist()))
            s = Structure(lattice=lat, species=species, coords=cur_frac,
                          coords_are_cartesian=False)
            info = {
                "method": "DiffCSP++",
                "lengths": cur_len.tolist(),
                "angles": cur_ang.tolist(),
                "volume": float(s.volume),
                "formula": s.composition.reduced_formula,
            }
            if output_dir is not None:
                cif_path = formula_dir / f"{formula}_{i + 1}.cif"
                CifWriter(s).write_file(str(cif_path))
                info["cif_path"] = str(cif_path)
            results.append((s, info))
        except Exception as e:
            print(f"  -> structure #{i + 1} build failed: {e}")

    return results
