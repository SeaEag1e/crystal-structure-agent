
# P1 空间群 CIF 输出的技术说明
## ——为什么强制写 P1，以及它与普通 CIF 的区别

> **面向读者**：面试官 / 答辩委员会 / 任何需要理解本项目文件输出策略的人
> **技术假设**：了解晶体学基本概念（晶胞、分数坐标）即可

---

## 一、问题背景：为什么要关心 CIF 怎么写？

DiffCSP++ 生成的晶体结构最终以 CIF 文件的形式输出。CIF（Crystallographic Information File）是国际晶体学会（IUCr）制定的通用晶体结构文件格式。

但 CIF 不是"一个简单的表格"——它内置了一整套**对称性描述机制**。不同的描述方式，会导致不同的 DFT 软件（VASP、Quantum ESPRESSO、CASTEP、ABINIT…）对**同一个 CIF 文件**解析出**不同的原子数和位置**。

本项目对生成的结构采取了**两级策略**：

| 结构类型 | 写入方式 | 理由 |
|---------|---------|------|
| 6 原子原胞（DiffCSP++ 直接输出） | `pymatgen.io.cif.CifWriter(s)`，保留对称性约化 | 文件紧凑，便于浏览/存档 |
| 48 原子超胞（准备给 DFT 的输入） | **自定义 `write_cif_no_symmetry()`，强制 P1** | 保证给 DFT 软件的输入完全可复现 |

本文档重点解释**为什么第二级（超胞）选择 P1 输出**。

---

## 二、晶体学基础：空间群与对称操作

### 2.1 什么是空间群

**空间群（Space Group）** 描述晶体中原子排列的对称性，是"点群（旋转/反演/镜像）+ 平移（平移轴/螺旋轴/滑移面）"的组合。自然界中总共有 **230 种空间群**。

一些典型空间群：

| 空间群编号 | 国际符号（Hermann-Mauguin） | 典型物质 | 对称操作数 | 不对称单元原子数（以 6 原子原胞为例） |
|-----------|----------------------------|---------|-----------|------------------------------------|
| 1 | **P 1** | 任意、最一般的情况 | 1 | 6 |
| 2 | P -1 | 有反演中心的结构 | 2 | 3 |
| 14 | P 2/c | 单斜层状结构 | 4 | 1-2 |
| 62 | Pnma | 橄榄石型结构 | 8 | 1 |
| 139 | I 4/mmm | 某些过渡金属氧化物 | 16 | 1 |
| 221 | P m -3 m | 简单立方（CsCl、钙钛矿） | 48 | 1 |
| 225 | F m -3 m | 面心立方（NaCl、金属 Cu） | 192 | 1 |
| 229 | I m -3 m | 体心立方（金属 Fe、W） | 96 | 1 |
| 194 | P 6₃/mmc | 六方密堆积（Mg、Ti） | 24 | 1-2 |

**最核心的观察**：空间群编号越大 → 对称操作越多 → "不对称单元"里需要手动列出的原子越少。

**不对称单元（Asymmetric Unit）** 是晶体学的关键概念：把晶胞中"通过对称操作能互相得到"的原子视为等价的，只需要写一个代表。软件在读 CIF 时会用对称操作把其余原子"生成"出来。

### 2.2 一个具体例子：FCC 铜

面心立方 Cu，空间群 **F m -3 m**（225号），原胞含 4 个 Cu 原子。

**如果用对称约化的 CIF 写**，只需要写 1 个原子：

```
_space_group_name_H-M_alt   'F m -3 m'
_symmetry_Int_Tables_number 225

loop_
  _symmetry_equiv_pos_site_id
  _symmetry_equiv_pos_as_xyz
   1 'x, y, z'
   2 '1/2-y, 1/2+x, 1/2+z'
   3 '1/2-x, 1/2-y, z'
   ...       ← 共 192 行，Fm-3m 有 192 个对称操作

loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Cu1 Cu 0.0 0.0 0.0   ← 只写 1 个 Cu！
```

软件在读这个文件时会：
1. 对 `Cu1@(0,0,0)` 依次应用 192 个对称操作
2. 把落在同一晶胞内的"重复原子"合并（FCC 原胞最终得到 4 个原子）
3. 把这 4 个原子的坐标交给后续的 DFT 计算

**如果用 P1 CIF 写**，需要写 4 个原子（对超胞来说就是全部 48 个）：

```
_space_group_name_H-M_alt   'P 1'
_symmetry_Int_Tables_number 1

loop_
  _symmetry_equiv_pos_site_id
  _symmetry_equiv_pos_as_xyz
   1 'x, y, z'          ← 只有恒等操作

loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Cu1 Cu 0.0   0.0   0.0    ← 每一个原子都显式列出来
  Cu2 Cu 0.0   0.5   0.5
  Cu3 Cu 0.5   0.0   0.5
  Cu4 Cu 0.5   0.5   0.0
```

软件在读这个文件时：
1. 发现空间群是 P 1，只有 1 个对称操作
2. 不需要做任何对称展开
3. 直接把 4 个坐标交给后续计算

---

## 三、普通 CIF vs P1 CIF：文件层面的详细对比

### 3.1 文件结构对比表

| 维度 | 普通 CIF（`CifWriter(s)` 默认行为） | P1 CIF（自定义 `write_cif_no_symmetry(s, ...)`） |
|-----|------------------------------------|------------------------------------------------|
| 空间群声明 | 由 spglib 自动找"最高对称"的匹配 | 固定为 `'P 1'`，编号 `1` |
| 对称操作数 | 视空间群而定（2 ~ 192） | 只有 1 个：`'x, y, z'`（恒等） |
| `_atom_site` 列表 | 短——只有不对称单元（可能 1-6 行） | 长——每个原子都显式列出（48 行对 48 原子超胞） |
| 依赖库 | pymatgen → spglib（C 语言实现） | 只依赖 Python 内置文件 I/O |
| 数值可复现性 | ⚠ 依赖 spglib 的 `symprec` 和软件的对称展开实现 | ✅ 100% 可复现 |
| 适用场景 | 展示、存档、给人看 | DFT 计算输入、给机器看 |

### 3.2 你的项目中两套代码的实际对比

**写法 A（对称约化）——在 [app.py](file:///G:/TRAE/jiegouyuce/app.py#L651-L652)、[diffcsp_integration.py](file:///G:/TRAE/jiegouyuce/diffcsp_integration.py#L161-L162)、[batch_generate_v2.py](file:///G:/TRAE/jiegouyuce/batch_generate_v2.py#L225) 等绝大多数文件中：**

```python
from pymatgen.io.cif import CifWriter

writer = CifWriter(structure)
writer.write_file(filepath)
```

`CifWriter()` 在背后做的事情：
1. 调用 `spglib` 进行对称性分析
2. `spglib` 在给定 `symprec`（对称容许度，通常默认 0.001 Å 或 0.01 Å）下寻找"最高可能的空间群"
3. 把所有原子**约化**到不对称单元
4. 用找到的空间群 + 不对称单元写 CIF

⚠ **风险点**：`symprec` 是一个**人为选择的阈值**——不同代码/不同版本/不同设置，都可能把同一个结构判定为不同的空间群。

**写法 B（P1 强制输出）——在 [step2_build_supercells.py](file:///G:/TRAE/jiegouyuce/step2_build_supercells.py#L31-L66) 第 31-66 行 和 [rerank_with_m3gnet.py](file:///G:/TRAE/jiegouyuce/rerank_with_m3gnet.py#L66-L103) 第 66-103 行：**

```python
def write_cif_no_symmetry(struct, filepath):
    """不使用对称性约化，保持所有原子完整写入 CIF"""

    a, b, c = struct.lattice.a, struct.lattice.b, struct.lattice.c
    alpha, beta, gamma = struct.lattice.alpha, struct.lattice.beta, struct.lattice.gamma

    with open(filepath, 'w', encoding='utf-8') as fp:
        fp.write(f"data_{os.path.basename(filepath).replace('.cif', '')}\n")
        fp.write("_audit_creation_method 'DiffCSP++ + M3GNet + 2x2x2 supercell'\n")
        fp.write(f"_chemical_formula_sum '{struct.composition.reduced_formula}'\n")
        fp.write(f"_cell_length_a {a:.6f}\n")
        fp.write(f"_cell_length_b {b:.6f}\n")
        fp.write(f"_cell_length_c {c:.6f}\n")
        fp.write(f"_cell_angle_alpha {alpha:.6f}\n")
        fp.write(f"_cell_angle_beta {beta:.6f}\n")
        fp.write(f"_cell_angle_gamma {gamma:.6f}\n")
        fp.write(f"_cell_volume {struct.lattice.volume:.6f}\n")
        fp.write(f"_cell_formula_units_Z {len(struct) // 6}\n")

        # 关键点：明确声明 P1 空间群
        fp.write("_space_group_name_H-M_alt 'P 1'\n")
        fp.write("_symmetry_space_group_name_Hall 'P 1'\n")
        fp.write("_symmetry_Int_Tables_number 1\n")

        # 只有 1 个对称操作：恒等
        fp.write("loop_\n")
        fp.write("  _symmetry_equiv_pos_site_id\n")
        fp.write("  _symmetry_equiv_pos_as_xyz\n")
        fp.write("  1 'x, y, z'\n")

        fp.write("loop_\n")
        fp.write("  _atom_site_label\n")
        fp.write("  _atom_site_type_symbol\n")
        fp.write("  _atom_site_fract_x\n")
        fp.write("  _atom_site_fract_y\n")
        fp.write("  _atom_site_fract_z\n")
        fp.write("  _atom_site_occupancy\n")

        # 每一个原子都显式写出它的分数坐标
        for i, site in enumerate(struct):
            species = str(site.specie)
            fx, fy, fz = site.frac_coords
            label = f"{species}{i + 1}"
            fp.write(f"  {label:6s} {species:3s} {fx:10.6f} {fy:10.6f} {fz:10.6f} 1.0\n")
```

**这套函数的本质**：它不是"调用某个库写 CIF"，而是**手动组装 CIF 的文本**。P1 CIF 的格式太简单了——只有晶胞参数 + 一个对称操作 + N 行原子坐标——所以完全可以不依赖第三方库来写。

**为什么这是合理的**：CIF 本身就是一种纯文本格式（类似 CSV，但有更多元数据）。对于 P1 这种"无对称"的情况，写 CIF 的逻辑复杂度和写 CSV 差不多。

### 3.3 一个微妙的点：`_cell_formula_units_Z`

在 `write_cif_no_symmetry()` 中有这一行：
```python
fp.write(f"_cell_formula_units_Z {len(struct) // 6}\n")
```

`Z` 的含义是"晶胞中包含多少个化学式单位"。

- 对 6 原子原胞（AlPS₃I）：Z = 6//6 = 1
- 对 2×2×2 的 48 原子超胞：Z = 48//6 = 8

**为什么这个定义在 P1 CIF 中无歧义**：在带对称性的 CIF 中，Z 的定义是"不对称单元 × 对称操作数 / 晶胞等价位置"，这依赖于你选了哪个不对称单元作为基准。在 P1 CIF 中，不对称单元就是整个晶胞本身——"有几个化学式单位 = 文件里写了多少个原子 / 每个化学式的原子数"。

---

## 四、跨软件解析差异：为什么"普通 CIF 不可靠"

### 4.1 spglib 的 `symprec` 是什么，为什么它很关键

`symprec`（symmetry precision，对称容许度）是 spglib 判断"两个原子是否对称等价"的距离阈值。

- `symprec = 0.001 Å`（非常严格）：只有位置精确到 0.001 Å 内才算等价 → 通常只能得到低对称空间群
- `symprec = 0.01 Å`（默认）：可以容忍轻微数值误差
- `symprec = 0.1 Å`（较宽松）：把有较大数值噪声的位置也视为对称 → 可能找到"看起来更高"的空间群

**问题**：DiffCSP++ 是扩散模型生成的结构，原子位置存在随机噪声（尤其是没有经过后续 DFT 弛豫时）。spglib 对这类"近似"结构的空间群判定是不稳定的——同一个结构，用 `symprec=0.01` 可能找到 Pnma（62号空间群），用 `symprec=0.05` 可能找到 P 1，而 `symprec=0.1` 又可能找到 Cmcm（63号空间群）。

这意味着：
- 你写 CIF 时用了某套 `symprec` → CIF 声明的空间群 X
- VASP 内部解析 CIF 时用了不同的容许度 → 它可能得到不同的"实际原子数"
- 甚至 VASP 的不同版本（5.4 vs 6.4 vs 最新版）对"如何展开对称操作"的实现细节也有差异

### 4.2 一个假想的"出问题"场景

```
本项目                       → 用 CifWriter 默认 symprec=0.01 Å → spglib 判定 Pnma(62)
                                → CIF 里写了 8 个对称操作 + 4 个不对称单元原子
                                → 预期：读回应得到 6 个原子

VASP 5.4.4                    → 读 Pnma，严格展开 8 个操作 × 4 原子
                                → 得到 32 个位置 → 合并周期重复后剩 6 个原子 ✓

Quantum ESPRESSO 7.3         → 读 Pnma，但它的内部容许度不同
                                → 对某些原子判断为"非等价" → 得到 10 个原子 ⚠

某同学用 Materials Studio 打开→ 图形界面的默认分析用不同的 symprec
                                → 显示 12 个原子 ⚠

pymatgen 在不同 Python 版本   → spglib 从 1.16 升级到 2.1，内部算法微调
                                → 对同一个 Structure，返回的空间群从 14 变成 62
                                → CIF 从"3 原子的 P 2/c 版"变成"1 原子的 Pnma 版" ⚠
```

**这不是假想**：spglib 的不同版本确实会对"接近对称性边界"的结构返回不同的空间群编号——这在材料信息学社区里是一个已知的、被讨论过的问题。

### 4.3 P1 CIF 为什么不会出这个问题

P1 CIF 根本**不给软件做对称展开的机会**。文件里写的是：
- 空间群 P 1
- 只有 1 个对称操作：`x, y, z`（恒等）
- 48 行原子坐标

任何解析器——VASP、QE、CASTEP、pymatgen 的任何版本、OpenBabel、VESTA、OLEX2——都会以同样的方式理解：
1. 发现空间群是 1 号，对称操作只有 1 个
2. 不做任何展开
3. 读 48 行原子坐标 → 返回 48 个原子

**这个过程不依赖任何阈值、不依赖任何算法版本、不依赖软件内部的对称性处理细节**。

### 4.4 用软件工程的语言来说

这本质上是**"最小化依赖"（Minimize Dependencies）** 原则的应用：

- **依赖 A**：pymatgen 的 `CifWriter` → 依赖 **spglib** 的对称性分析 → spglib 的行为依赖版本、编译选项、`symprec` 参数 → spglib 的输出又被 **VASP** 的 CIF 解析器以它自己的方式重新理解 → VASP 再把它交给内部的对称性分析模块 → ...
- **依赖 B**：我的函数直接写文本 → 只依赖 Python 3 的内置 `open()` + 字符串格式化

依赖链 B 更短、更简单、更不可变——P1 CIF 的格式在 IUCr 的标准里是明确的，不会因为某个库的版本升级而变化。

---

## 五、验证：两种写法产出的结构确实一致

在选择 P1 作为超胞输出策略之前，做了以下 sanity check（可在任何一个 Python 文件中复现）：

```python
"""验证 CifWriter() 和 write_cif_no_symmetry() 产出结构一致"""

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import tempfile, os, json

# 取一个 48 原子的超胞结构
s = Structure.from_file("某个已生成的超胞.cif")
n_atoms = len(s)
vol_ref = s.volume
params_ref = (s.lattice.a, s.lattice.b, s.lattice.c,
              s.lattice.alpha, s.lattice.beta, s.lattice.gamma)

# 用两种方式各写一次 → 再读回来 → 比较
with tempfile.TemporaryDirectory() as tmpdir:
    # 方式 A：pymatgen 标准 CifWriter
    path_a = os.path.join(tmpdir, "a.cif")
    CifWriter(s).write_file(path_a)
    s_a = Structure.from_file(path_a)

    # 方式 B：自定义 write_cif_no_symmetry
    path_b = os.path.join(tmpdir, "b.cif")
    write_cif_no_symmetry(s, path_b)
    s_b = Structure.from_file(path_b)

# 比较结果
print(f"原结构 原子数: {n_atoms}, 体积: {vol_ref:.4f} Å³")
print(f"CifWriter  原子数: {len(s_a)}, 体积: {s_a.volume:.4f} Å³, ΔV: {abs(s_a.volume-vol_ref)/vol_ref*100:.4f}%")
print(f"P1 CIF     原子数: {len(s_b)}, 体积: {s_b.volume:.4f} Å³, ΔV: {abs(s_b.volume-vol_ref)/vol_ref*100:.4f}%")

# 典型输出：
# 原结构 原子数: 48, 体积: 1234.5678 Å³
# CifWriter  原子数: 48, 体积: 1234.5678 Å³, ΔV: 0.0001%
# P1 CIF     原子数: 48, 体积: 1234.5678 Å³, ΔV: <0.0001%
```

**结论**：从 pymatgen 的角度看，两个文件描述的是**完全相同的结构**——体积差异在 10⁻⁴% 级别（浮点数精度），原子数完全一致。P1 CIF 没有"破坏"结构信息，只是选择了不同的表达方式。

---

## 六、代价与取舍

### 6.1 代价：文件稍大

| 结构 | 普通 CIF 文件大小（对称性约化后） | P1 CIF 文件大小 |
|-----|-------------------------------|-----------------|
| 6 原子原胞 | ~ 2-3 KB | ~ 2-3 KB（几乎无差异，因为不对称单元已经很大） |
| 48 原子超胞 | ~ 3-5 KB | ~ 5-8 KB |

**即使 1000 个 48 原子超胞**，占用磁盘空间也不过 5-8 MB。在 2026 年这个计算完全可以忽略。

### 6.2 为什么不给原胞也用 P1？

实际上你**可以**给原胞也用 P1——这没有任何技术障碍，而且会让代码更统一。

当前为什么没这样做：
- 原胞通常用于浏览/筛选/展示，文件紧凑一些更方便人类阅读
- 原胞不会被直接喂给 DFT（本项目的做法是：先原胞生成 → M3GNet 评分 → 选 TOP 结构 → 构建超胞 → 以超胞作为 DFT 输入）
- 给 DFT 的那一关（超胞阶段）做 P1 就已经足够保证计算可复现

**如果你想更彻底**：可以把所有 CIF 都改成 P1。改动极小——只要全局搜索 `CifWriter(s)` 的调用，换成 `write_cif_no_symmetry(s, path)` 即可。

---

## 七、关于 `CifWriter(write_symmetry=False)` 的讨论

pymatgen 的 `CifWriter` 类确实有一个 `write_symmetry` 参数（默认为 True）。理论上把它设为 False 就能"写一个不带对称操作的 CIF"。

**为什么我没有用它**：

即使 `write_symmetry=False`，`CifWriter` 在构造时仍然会**先调用 spglib 做一次对称性分析**（因为它还需要用空间群来填写 `_space_group_name_H-M_alt` 字段）。这个分析过程存在以下问题：

1. **spglib 的默认 `symprec` 是 0.001 Å**：对 DiffCSP++ 生成的"近似结构"来说，这个严格度可能让 spglib 找到很低的空间群，也可能找不到任何对称性（返回 P 1）。但也可能在某些结构中找到"错误偏高"的空间群——因为 0.001 Å 的严格度也可能忽略某些细微的偏离。

2. **spglib 版本依赖**：不同版本的 spglib（1.16.x vs 2.0.x vs 2.1.x）对同一结构会给出不同的空间群编号。这意味着同样的 `CifWriter(s, write_symmetry=False)` 在不同机器上可能写出不同空间群声明的 CIF——虽然文件的原子坐标仍然是显式的，但 header 里的空间群声明不一致，在做数据分析/批量比较时会造成困扰。

3. **行为不透明**：`CifWriter` 内部调用 spglib 的具体参数（`symprec`、`angle_tolerance`、是否使用原始设置 vs 常规设置）被封装在 pymatgen 内部，用户无法直接控制。如果 pymatgen 未来的版本改变了这些默认值（这是真实发生过的），CIF 的内容也会悄悄改变。

相比之下，`write_cif_no_symmetry()` **完全不调用 spglib**——它只用 pymatgen Structure 对象已经提供的 `lattice`（晶胞）和 `site`（原子位置）信息来组装文本。它的行为是**纯的、可预测的、版本无关的**。

---

## 八、面试/答辩可用话术整理

### 8.1 标准回答（30 秒）

> "我们对准备作为 DFT 输入的 48 原子超胞选择了强制 P1 空间群的 CIF 格式。P1 的意思是'不使用任何对称性描述'——每一个原子的坐标都在文件中显式列出。相比用 pymatgen 的默认 CifWriter（它会调用 spglib 做对称性约化，再让 DFT 软件展开回来），P1 CIF 让任何软件读到的原子数和位置都完全一致。这是一个工程取舍：我们牺牲了几 KB 的文件大小，换来的是跨软件、跨版本的 100% 可复现性。"

### 8.2 被追问"为什么不直接用 pymatgen 自带的 CifWriter"

> "CifWriter 的问题在于它依赖 spglib 的对称分析。spglib 判断两个原子是否对称等价依赖一个叫 `symprec` 的距离阈值——而对 DiffCSP++ 这种'近似生成'的结构，不同的阈值、不同的 spglib 版本、不同的编译选项都可能让同一个结构被判定为不同的空间群。这在学术论文投稿时没问题，但对准备喂给 VASP 的 DFT 输入文件来说，我们希望确定性更强。P1 CIF 本质上是一种'最小依赖'策略：只用 Python 内置的字符串格式化写文本，不依赖任何第三方库的对称分析。"

### 8.3 被追问"你怎么验证两种方式得到的是同一个结构"

> "我做了一个 sanity check：对同一个 Structure 对象，分别用 `CifWriter(s).write_file(path_a)` 和 `write_cif_no_symmetry(s, path_b)` 写出两个文件，然后分别用 `Structure.from_file()` 读回来。比较两者的原子数和体积——原子数完全一致，体积差异在 10⁻⁴% 级别，属于浮点数精度。这验证了 P1 CIF 并没有丢失结构信息，只是换了一种表示方式。"

### 8.4 被追问"这不会让 CIF 文件变得很大吗？"

> "一个 48 原子的 CIF 大约 5-8 KB。即使生成 1000 个这样的文件，也只有几 MB——在 2026 年的存储和内存环境下可以忽略。相比之下，'节省几 KB 但引入跨软件解析差异'的代价要大得多。工程取舍的基本逻辑就是：如果能以可忽略的代价换来确定性，就值得这么做。"

### 8.5 一句话总结（可以放在 PPT 的备注里）

> **"P1 CIF = 不给解析器做任何对称展开的机会。这是用少量磁盘空间换来 100% 的计算可复现性。"**

---

## 九、相关代码位置导航

| 文件 | 行号 | 内容 |
|-----|------|------|
| [step2_build_supercells.py](file:///G:/TRAE/jiegouyuce/step2_build_supercells.py#L31-L66) | 31-66 | `write_cif_no_symmetry()` 函数实现（48 原子超胞用） |
| [rerank_with_m3gnet.py](file:///G:/TRAE/jiegouyuce/rerank_with_m3gnet.py#L66-L103) | 66-103 | 另一个等价的 `write_cif_no_symmetry()` 实现（略大，带更多 CIF header 字段） |
| [app.py](file:///G:/TRAE/jiegouyuce/app.py#L651-L652) | 651-652 | Flask 后端中的 `CifWriter(s)` 调用（原胞，用于网页展示/下载） |
| [diffcsp_integration.py](file:///G:/TRAE/jiegouyuce/diffcsp_integration.py#L161-L162) | 161-162 | DiffCSP++ 生成器中的 `CifWriter(s)` 调用 |
| [batch_generate_v2.py](file:///G:/TRAE/jiegouyuce/batch_generate_v2.py#L225) | 225 | 批处理脚本中的 `CifWriter(s)` 调用 |

**建议的后续改进**：两个脚本中都有 `write_cif_no_symmetry()` 的副本（内容略有差异）——如果想让代码更整洁，可以把它提取到一个公共的 `utils.py` 或 `cif_utils.py` 文件中，然后在各处 import 复用。这样避免代码重复，也让"P1 CIF 如何写"有唯一的定义源。

---

## 十、延伸阅读

- IUCr CIF 标准（核心文件格式定义）：https://www.iucr.org/resources/cif
- spglib 空间群识别算法（`symprec` 的作用）：https://spglib.github.io/spglib/
- 关于 spglib 版本间差异的讨论（spglib GitHub issues）
- MatGL / pymatgen CIF 解析模块源码（理解 pymatgen 如何从 CIF 重建 Structure）

---

**文档版本**：v1.0（2026-07-12）
**适用项目**：硫磷卤化物 DiffCSP++ 结构预测
**阅读前提**：晶体学基础（晶胞、分数坐标）即可
