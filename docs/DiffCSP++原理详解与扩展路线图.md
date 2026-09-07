
# DiffCSP++ 原理详解与扩展路线图
## ——从扩散模型到晶体结构预测，再到多属性性能预测

> **定位**：面向希望深入理解 DiffCSP++ 内部机制，并在此基础上扩展新功能（如带隙预测、体模量预测等）的开发者和研究者
> **适用读者**：已经掌握项目基本使用方法，希望深入源码并做二次开发的人
> **前置知识**：扩散模型（DDPM）基本概念、图神经网络（GNN）基本概念、晶体学基础（晶胞、倒易空间、分数坐标）

---

## 第一部分：DiffCSP++ 原理详解

### 1.1 核心思想：在"晶体空间"中做反向扩散

DiffCSP++（*Composition-conditioned Structure Prediction*）的核心思想可以用一句话概括：

> **学会真实晶体的概率分布，然后从这个分布中采样，生成新的晶体结构。**

传统 DDPM（Denoising Diffusion Probabilistic Models）是在图像像素空间中做正向加噪、反向去噪。DiffCSP++ 把这个思想移植到了晶体学空间中：

| DDPM（图像） | DiffCSP++（晶体） |
|-------------|-----------------|
| 数据维度：(H, W, 3) | 数据维度：{frac_coords(N, 3), lattice(3, 3), atom_types(N)} |
| 加噪对象：像素值 | 加噪对象：原子分数坐标 + 晶胞基矢矩阵 |
| 噪声分布：高斯 N(0, σ²I) | 坐标噪声：包裹正态 *Wrapped Normal*（周期性）；晶格噪声：普通高斯 |
| 反向过程：从纯噪声 → 逐步去噪 → 真实图像 | 反向过程：从随机坐标/晶格 → 逐步去噪 → 合理晶体结构 |

**为什么要引入"包裹正态"（Wrapped Normal）？**

晶体中的原子坐标是**分数坐标**（fractional coordinates），取值范围是 [0, 1)，具有周期性边界条件（PBC）。如果对坐标直接加普通高斯噪声，原子可能跑到 [0, 1) 之外——这在物理上是不合理的（它其实只是相邻晶胞里的同一个原子）。

DiffCSP++ 的做法是：

```
x_t = (x_0 + σ_t · ε) mod 1,    ε ~ N(0, I)
```

即把坐标加噪后再 mod 1（对 1 取余），保证它始终在 [0, 1) 内。这样噪声分布是"包裹正态"分布：

```
p_wrapped(x; σ) = Σ_{k=-∞}^{∞} (1/√(2πσ²)) · exp(-(x-k)² / 2σ²)
```

代码中对应的实现是 `diff_utils.py` 的 `d_log_p_wrapped_normal()`，它计算的是这个分布的对数导数（用于训练目标）。

### 1.2 前向加噪过程（Training）

前向过程（training 时用到）是把真实晶体逐步混入噪声，公式如下：

```
l_t = √(ᾱ_t) · l_0 + √(1-ᾱ_t) · ε_l,    ε_l ~ N(0, I)    # 晶格加噪
x_t = (x_0 + σ_t · ε_x) mod 1,              ε_x ~ N(0, I)    # 坐标加噪（包裹正态）
```

其中：
- `α_t = 1 - β_t`，`β_t` 是由 `BetaScheduler` 产生的递增方差 schedule（从 0.0001 缓慢增加到某个值，用 cosine 模式）
- `ᾱ_t = ∏_{s=1}^t α_s`，即累积乘积（用于跳步采样）
- `σ_t` 是由 `SigmaScheduler` 产生的坐标噪声强度，从 `sigma_begin=0.005` 指数增长到 `sigma_end=0.5`

从源码 `diffusion.py` 的 `forward()` 可以看到它的实际执行：

```python
# 1. 采样时间步 t ~ Uniform(1..T)
times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)

# 2. 计算 ᾱ_t（用于跳步加噪：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε）
alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
c0 = torch.sqrt(alphas_cumprod)     # √ᾱ_t
c1 = torch.sqrt(1 - alphas_cumprod) # √(1-ᾱ_t)

# 3. 晶格加噪（普通高斯）
rand_l = torch.randn_like(lattices)
input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

# 4. 坐标加噪（包裹正态：加噪后 mod 1）
sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
rand_x = torch.randn_like(frac_coords)
input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

# 5. 送入解码器，预测噪声
pred_l, pred_x = self.decoder(time_emb, batch.atom_types,
                               input_frac_coords, input_lattice,
                               batch.num_atoms, batch.batch)

# 6. 训练损失：预测的噪声 ↔ 真实的噪声
tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)
loss_lattice = F.mse_loss(pred_l, rand_l)
loss_coord   = F.mse_loss(pred_x, tar_x)
loss = cost_lattice * loss_lattice + cost_coord * loss_coord
```

**关键点说明**：
- 晶格用标准 DDPM 加噪（线性组合）
- 坐标用"包裹正态"加噪（mod 1），损失项不是简单的 MSE(ε, ε̂)，而是 MSE(d_log_p_wrap(ε), pred_x)——这是一个重要的工程细节
- `sigmas_norm` 是对梯度做归一化的量，防止不同时间步损失尺度不一致

### 1.3 反向采样过程（Generation）

采样时，从完全随机的起点开始，逐步去噪 T=1000 步。DiffCSP++ 使用的是 **Predictor-Corrector** 采样策略，即每一步包含：

1. **Corrector 步**（朗之万校正）：沿对数似然梯度做小步微调，增加样本多样性
2. **Predictor 步**（标准 DDPM 反向去噪）：从 t 走到 t-1

从源码 `diffusion.py` 的 `sample()` 方法看：

```python
# ===== 初始化：t=T 时的随机起点 =====
l_T = torch.randn([batch_size, 3, 3]).to(device)    # 随机晶格
x_T = torch.rand([batch.num_nodes, 3]).to(device)    # 均匀随机坐标（因为分数坐标在[0,1]）

# ===== 从 t=T 走到 t=1 =====
for t in range(T, 0, -1):
    # --- (a) Corrector 步：沿梯度走一小步，加噪声 ---
    # step_size = learning_rate * (sigma_t / sigma_begin)^2 （信噪比越大步长越大）
    # pred_x = model(x_t, l_t, t)  预测分数坐标的"方向"
    # x_{t-0.5} = x_t - step_size * pred_x + std * rand_x
    # 注意：晶格在我们用的 mp_csp 模型里 keep_lattice = False 吗？
    #       实际上在 mp_csp 模型里，cost_lattice=1.0, cost_coord=1.0，都在优化

    # --- (b) Predictor 步：标准 DDPM 去噪 ---
    # x_{t-1} = (x_t - step_size * pred_x) + std_x * rand_x
    # l_{t-1} = (1/√α_t) * (l_t - ((1-α_t)/√(1-ᾱ_t)) * pred_l) + σ_t * rand_l
    # （这个公式就是 DDPM 的反向过程）

    traj[t-1] = {frac_coords: x_{t-1} % 1, lattices: l_{t-1}}

return traj[0], traj  # 返回 t=0 的结果和完整轨迹
```

**为什么 Predictor-Corrector 比纯 DDPM 更好？**
- DiffCSP++ 学到的是 score（对数概率的梯度 ∇_x log p_t(x)），而不是直接预测 x_0
- 朗之万动力学（Langevin dynamics）利用这个 score 把样本往高密度区域（即"更像真实晶体"的区域）推
- 预测器负责"大致去噪"，校正器负责"局部精调"
- 对于晶体这种结构敏感性强的分布，PC 采样比纯 DDPM 能得到更合理的局部结构

### 1.4 CSPNet——解码器（神经网络架构）

解码器是真正学习"从噪声到晶体"映射的核心部分。它的架构是一个**图神经网络（GNN）**，叫 CSPNet（在 `cspnet.py` 中）。

**输入**：
- `atom_types` (N,)：原子类型索引（1=H, 2=He, ...）
- `frac_coords` (N, 3)：加噪后的分数坐标
- `lattices` (batch_size, 3, 3)：加噪后的晶格基矢矩阵
- `time_emb` (batch_size, time_dim)：扩散时间 t 的嵌入
- `num_atoms` (batch_size,)：每个晶体的原子数
- `node2graph` (N,)：节点到图的映射（即 batch vector）

**输出**：
- `pred_x` (N, 3)：坐标方向的"预测梯度"
- `pred_l` (batch_size, 3, 3)：晶格方向的"预测梯度"

**架构详解**：

```
步骤 1：节点嵌入
   atom_types → Embedding(100, 512) → h_i(512)
   t → SinusoidalTimeEmbedding(256) → t_emb(256)
   拼接 h_i 与 t_emb（每个原子重复同样的 t_emb） → h_i_new(512)
```

```
步骤 2：生成图连接（edge_style="fc" 表示全连接）
   对每个晶体的 N 个原子，构造 N×N 的全连接图
   edges = {(i, j) | 1 ≤ i, j ≤ N}
   frac_diff_ij = (x_j - x_i) mod 1 （边特征：相对坐标向量）

   edge_feature_ij = MLP([h_i || h_j || (lattices^T lattices)_flatten || Sinusoids(frac_diff_ij)])
   其中 Sinusoids 是把分数坐标差 (3维) 映射到高频正弦基 (3 × 128 × 2 = 768维)
   这让模型能学到周期性边界下的距离/方向信息
```

```
步骤 3：消息传递（6 层 CSPLayer）
   每一层做：
     (a) 消息聚合：agg_i = mean_j edge_feature_ij
     (b) 节点更新：h_i_new = node_MLP([h_i || agg_i])
     (c) 残差连接：h_i ← h_i + h_i_new
     (d) LayerNorm（可选）
```

```
步骤 4：输出头
   (a) 坐标输出：CoordOut = Linear(512, 3, bias=False) → per-atom 3维向量
       pred_x = CoordOut(h_final)

   (b) 晶格输出：GraphPool(h_final) → graph_vec(512)
       LatOut = Linear(512, 9, bias=False) → (batch, 9) → reshape → (batch, 3, 3)
       pred_l = pred_l @ lattices   # 预测"修正量"然后与当前晶格相乘（IP = inner product 模式）
       （@ = 矩阵乘法；预测一个 3×3 的"修正矩阵"而不是直接预测新晶格，更稳定）
```

**为什么这是一个好架构？**
1. **全连接图**：对最大 20 原子的小晶体是合理的（O(N²) 边可以接受），所有原子间都能传消息，没有截断距离带来的信息损失
2. **显式晶格特征**：把 `lattices^T lattices`（即晶体的度量张量 g_ij = a_i · a_j，它在倒易变换下是不变的）作为边特征的一部分，让模型"看到"晶胞的形状和大小
3. **正弦位置编码**：frac_diff 经过 Sinusoids 处理后，模型能感知原子间的相对位置——这个设计来自 NLP 中的 Transformer，但在晶体学中同样有效，因为它把 [0,1) 的连续坐标变成了丰富的高频/低频特征
4. **残差 + LayerNorm**：标准的深度网络训练技巧，让 6 层消息传递也能稳定训练

### 1.5 "Composition-conditioned"是怎么实现的？

这是 DiffCSP++ 区别于其他"无条件"扩散模型的关键：

**在训练时**，模型始终能看到真实的原子类型（`atom_types`），它学到的是：

```
p(x, lattice | atom_types)
```

即"给定这堆元素，合理的晶体结构是什么分布"。这是通过把 atom_types 直接嵌入到节点特征中实现的——模型从第一步就知道"这个位置是 Al，那个是 S"。

**在生成时**，用户指定化学式（比如 AlPS3I），它被解析成 atom_types 列表 [13, 15, 16, 16, 16, 53]，然后把这个列表喂给 CSPNet。模型从纯随机起点开始去噪，每一步都"知道"要去的终点必须具有这 6 个特定原子，所以会把它们推到符合这个化学式的合理几何位置上。

**为什么这比"不考虑化学式"的生成更有用？**
- 在材料发现中，你通常已经知道想要什么元素，只想知道它们怎么排列
- 条件生成把搜索空间从"所有可能的晶体"缩小到"化学式固定的结构"

### 1.6 时间步 T=1000 步是什么概念？

| 指标 | 值 | 说明 |
|-----|-----|------|
| 总步数 | T=1000 | 与图像扩散模型（如 Stable Diffusion）一样的量级 |
| 时间嵌入维度 | 256 | Transformer 标准的 sin/cos 位置编码 |
| 每步时间 | ~0.001-0.01 秒/GPU | 取决于 GPU 型号 |
| 生成 1 个 6 原子结构总时间 | ~5-30 秒 | 主要瓶颈是 1000 步循环，无法并行化 |
| 批量生成 | 可并行 | 把 batch 设为 K，一次生成 K 个结构，几乎是免费的并行 |

---

## 第二部分：项目当前架构与数据流

### 2.1 总体架构图

```
        ┌────────────────────────────┐
        │   用户输入 (化学式)         │
        │   e.g., "AlPS3I"            │
        └────────┬───────────────────┘
                 │
        ┌────────▼──────────────────────────────────┐
        │  diffcsp_integration.py                   │
        │  ┌────────────────────────────────────┐  │
        │  │chemparse.parse_formula("AlPS3I")   │  │
        │  │  → {'Al':1, 'P':1, 'S':3, 'I':1}   │  │
        │  │→ atom_types = [13, 15, 16, 16, 16, 53]│  │
        │  │→ Data(atom_types, num_atoms=6)      │  │
        │  │→ Batch.from_data_list([...])        │  │
        │  │→ model.sample(batch, step_lr=1e-5)  │  │
        │  │  （1000步 PC 采样循环）              │  │
        │  └──────────┬───────────────────────────┘  │
        │             │ outputs dict                 │
        │             ▼                              │
        │  outputs['frac_coords'] → frac coords     │
        │  outputs['lattices']   → 3×3 matrices     │
        │  outputs['atom_types'] → argmax → indices│
        │             │                              │
        │             ▼                              │
        │  lattices_to_params_shape → (a,b,c,α,β,γ) │
        │  → pymatgen Structure objects              │
        │  → 保存 CIF 文件                           │
        └──────────────────┬─────────────────────────┘
                           │
               ┌───────────▼──────────────┐
               │ pymatgen Structure list   │
               │ （每个结构一个对象）      │
               └───────────┬──────────────┘
                           │
        ┌──────────────────▼──────────────────────┐
        │ app.py: 多模型打分 & Web 服务            │
        │                                          │
        │   predict_formation_energy(structure):    │
        │     1. 尝试 M3GNet-Eform-MP-2019.4.1     │
        │        → model.predict_structure(s)       │
        │     2. 失败 → 尝试 M3GNet-MP-2021.2.8-EFS │
        │     3. 失败 → 尝试 MEGNet-MP-2018.6.1-Eform│
        │     4. 全部失败 → fallback: 元素焓求和     │
        │                                          │
        │   返回：eV/atom（负值=稳定）              │
        └───────────────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │  前端展示结果   │
                  │  - 能量排名表   │
                  │  - Three.js 3D  │
                  │  - CIF 下载     │
                  └──────────────────┘
```

### 2.2 关键模块间的耦合关系

| 模块 | 职责 | 依赖 | 对外 API |
|-----|------|------|---------|
| `diffcsp_integration.py` | DiffCSP++ 模型加载 + 条件生成 | `diffcsp` 源码包、`torch_geometric`、`chemparse`、`pymatgen` | `generate_structures_diffcsp(formula, num_structures=3)` |
| `app.py` | Flask 路由 + 结构生成调度 + M3GNet 打分 | `flask`、`matgl`、`pymatgen` | Flask routes (`/`, `/predict`, `/preview/...`)；内部函数 `predict_formation_energy()`, `get_model()` |
| `templates/index.html` | 前端 UI | HTML/JS/Three.js | 浏览器端交互 |

**设计模式**：
- 生成器与打分器解耦——两者通过 pymatgen Structure 对象通信，互不关心内部实现
- 打分器使用 **fallback chain**（降级链），保证最大可用性
- 模型缓存机制：首次加载慢，后续复用极快（这对 Flask 服务尤为重要，避免每次请求都加载 1GB 的模型权重）

### 2.3 输入/输出格式速查

**CSPNet 输入的 Batch 对象**：
```python
# Data 对象
Data(
    atom_types  = tensor([13, 15, 16, 16, 16, 53], dtype=torch.long),  # 原子序数列表
    num_atoms   = 6,
    num_nodes   = 6  # 与 num_atoms 相同（Pytorch Geometric 的约定）
)

# Batch = [Data1, Data2, ..., DataK] 打包后的图结构
# batch.atom_types: (K·N,)
# batch.num_atoms: (K,)
# batch.batch: (K·N,)  # batch vector，0..0, 1..1, ..., K-1..K-1
```

**`model.sample()` 的返回**：
```python
outputs = {
    'frac_coords': tensor(N·K, 3),          # 分数坐标 [0, 1)
    'atom_types': tensor(N·K,),             # 原子类型（argmax 后的索引）
    'num_atoms':  tensor(K,),                # 每个结构的原子数
    'lattices':   tensor(K, 3, 3)            # 每个结构的晶格基矢矩阵
}
```

**转为 pymatgen Structure**：
```python
species = [chemical_symbols[int(z)] for z in atom_types]
lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
structure = Structure(lattice=lat, species=species, coords=frac_coords, coords_are_cartesian=False)
```

---

## 第三部分：扩展性能预测功能的详细路线图

### 3.1 当前只能预测什么？

在你当前的代码中，`app.py` 的 `predict_formation_energy()` 只能预测一个属性：

- **形成能（Formation Energy）**：eV/atom，衡量化合物的热力学稳定性

如果你想预测其他材料属性（带隙、体模量、剪切模量、折射率……），需要扩展。下面给出三个层次的扩展方案。

### 3.2 方案一：用 matgl 已有预训练模型（最快，1 小时搞定）

matgl 官方已经提供了几个属性预测模型。**最简单的扩展就是：照抄 `predict_formation_energy()` 的模式，多写几个类似的函数。**

#### matgl 官方可用的预训练模型（截至 2026年）

| 模型名 | 预测属性 | 单位 | 架构 | 数据来源 | 典型精度 |
|-------|---------|------|------|---------|---------|
| `M3GNet-Eform-MP-2019.4.1` | 形成能 | eV/atom | M3GNet | MP DFT | MAE ≈ 0.02 |
| `M3GNet-MP-2021.2.8-EFS` | 能量 + 力 + 应力 | eV/atom, eV/Å, eV/Å³ | M3GNet | MP DFT Trajectory | 高，通用势 |
| `MEGNet-MP-2018.6.1-Eform` | 形成能 | eV/atom | MEGNet | MP | MAE ≈ 0.03 |
| `MEGNet-MP-2019.4.1-BandGap-mfi` | 带隙（GGA-PBE 级别） | eV | MEGNet | MP | MAE ≈ 0.2-0.3 |
| `CHGNet-MPtrj-2023.12.1-PES` | 能量/力/应力/磁矩 | 多种 | CHGNet | MP Trajectory | 高 |
| `M3GNet-MP-2021.2.8-PES` | 能量/力/应力 | 多种 | M3GNet | MP Trajectory | 高 |
| `TensorNet-PES-MatPES-PBE-2025.2` | 能量/力/应力 | 多种 | TensorNet | MatPES | 最新 SOTA 之一 |

**说明**：
- 以 `EFS` / `PES` 结尾的是**势能模型**（能预测能量/力/应力，可用于结构弛豫），也可以单独用它的能量作为"形成能"的替代
- 以 `Eform` / `BandGap` 结尾的是**属性预测模型**（只输出单一标量）

#### 在 app.py 中新增带隙预测的示例代码

新增一个函数 `predict_band_gap()`，完全仿照 `predict_formation_energy()` 的模式：

```python
# ============ 新增代码块：带隙预测 ============
_BANDGAP_MODEL = None
_BANDGAP_LOCK = threading.Lock()

def get_bandgap_model():
    """单例加载带隙预测模型。"""
    global _BANDGAP_MODEL
    if _BANDGAP_MODEL is None and MATGL_AVAILABLE:
        with _BANDGAP_LOCK:
            if _BANDGAP_MODEL is None:
                # 目前 matgl 有 MEGNet-MP-2019.4.1-BandGap-mfi 模型
                # 未来 M3GNet 带隙模型若发布，可添加到降级链中
                for model_name in ["MEGNet-MP-2019.4.1-BandGap-mfi"]:
                    try:
                        print(f"加载 {model_name} ...")
                        _BANDGAP_MODEL = matgl.load_model(model_name)
                        return _BANDGAP_MODEL
                    except Exception as e:
                        print(f"  加载失败: {e}")
                        continue
                print("警告：带隙模型不可用，将返回 None")
    return _BANDGAP_MODEL

def predict_band_gap(structure):
    """预测 PBE 级别的带隙（单位：eV）。
    返回 None 表示模型不可用/推理失败。"""
    model = get_bandgap_model()
    if model is None:
        return None
    try:
        result = model.predict_structure(structure)
        return round(float(result), 4)
    except Exception as e:
        print(f"[BandGap 推理失败] {e}")
        return None
```

#### 在 `/predict` 路由中加入带隙信息

修改 `/predict` 路由，对每个结构额外调用 `predict_band_gap()`：

```python
# 在 /predict 路由原有的 for 循环中加入：
predictions.append({
    'formula': s.composition.reduced_formula,
    'key': key,
    'index': i + 1,
    'method': 'DiffCSP++ + M3GNet',
    'formation_energy': eform,
    'band_gap_ev': predict_band_gap(s),   # ← 新增
    'stability': '稳定' if (eform is not None and eform < 0.0) else '不稳定',
    'lengths': info['lengths'],
    'angles': info['angles']
})
```

#### 前端展示修改

在 `templates/index.html` 的结果表格中新增一列：

```html
<th>带隙 (eV)</th>
...
<td>{{ pred.band_gap_ev if pred.band_gap_ev is not none else '-' }}</td>
```

**这套方案的优点**：一天之内就能上线，无需训练任何模型，直接利用 matgl 官方预训练权重。

**缺点**：matgl 没有为硫磷卤化物专门优化，带隙/力学性能的精度可能一般；且属性数量有限，无法预测更偏应用的指标（如离子电导率、光学吸收等）。

### 3.3 方案二：DiffCSP++ 内部的属性预测头（中等难度，1-2 周）

如果你对 DiffCSP++ 生成的结构本身做"生成即打分"有更高的期望，可以在 CSPNet 上加一个**标量预测头**（scalar head）。

查看源码 `cspnet.py`，你会发现 CSPNet 本身就预留了这个机制：

```python
# cspnet.py 中已有的代码片段
self.pred_scalar = pred_scalar    # 布尔参数，默认 False
if self.pred_scalar:
    self.scalar_out = nn.Linear(hidden_dim, 1)

# forward() 的末尾：
if self.pred_scalar:
    return self.scalar_out(graph_features)  # 直接输出一个标量
```

也就是说，DiffCSP++ 的作者已经内置了"用图表示向量预测标量属性"的钩子。你只需要：

**步骤 1：数据准备**
- 需要一个晶体 → 目标属性的数据集（比如从 Materials Project 下载带隙、体模量等）
- 数据格式：`[(Structure, target_value), ...]`

**步骤 2：训练**
- 加载预训练的 DiffCSP++ 权重作为 backbone（迁移学习）
- 冻结或微调前几层 CSPLayer，新增一个 `pred_scalar=True` 的输出头
- 用 MSE Loss 训练这个预测头
- 训练数据：选 10 万+ Materials Project 结构，训练 10-20 epochs 即可收敛

**步骤 3：修改 diffcsp_integration.py**
- 在 `generate_structures_diffcsp()` 的返回值中额外返回预测的标量属性
- 不再需要 M3GNet 做"二次打分"，DiffCSP++ 生成时就附带了能量/带隙估计

**这种方案的优点**：
- 生成与评估端到端，速度更快（无需加载第二个模型）
- 可以联合优化"让生成出的结构的预测能量更低"——这是主动学习 / 强化学习的方向

**缺点**：
- 需要 GPU 训练资源和时间
- 预训练权重可能需要从 matgl 适配到 DiffCSP++ 的格式
- 不如 M3GNet 在能量预测上准确（M3GNet 专门做了三体相互作用的显式建模）

### 3.4 方案三：真正的多属性统一预测（推荐长期目标，1-3 个月）

这是最严谨、可扩展性最强的方案：**把每个结构都送到 M3GNet / CHGNet / TensorNet 的势能模型做一次性"能量 + 力 + 应力 + 派生属性"的联合预测。**

核心思想是：M3GNet / CHGNet 这类 PES 模型（势能面模型）能预测每个结构的总能量、每个原子的受力、晶胞的应力张量——然后从这些量中**派生**出更多的宏观属性：

| 目标属性 | 如何从 PES 模型的输出派生 | 现有工具 |
|---------|--------------------------|---------|
| 形成能 E_f | E_total(struct) - Σ E(element_i) | `model.predict_structure()` |
| 带隙 E_g | 无法从 PES 直接得到 → 需单独的带隙模型 | `MEGNet-MP-2019.4.1-BandGap-mfi` |
| 体模量 B | -V · d²E/dV²（对体积做 EOS 拟合） | 计算 E(V) 曲线 + Birch-Murnaghan 拟合 |
| 剪切模量 G | 从弹性刚度矩阵 C_ij 派生 | 需要 6 个不同应变 + 应力响应 |
| 泊松比 ν | (3B - 2G) / (6B + 2G) | 上两项派生 |

具体实现时，建议新增一个模块文件 `predict_properties.py`：

```python
"""
predict_properties.py ——统一接口，对结构做多属性预测
用法：
   results = predict_all_properties(structure)
   # results: {'eform_ev_per_atom': -0.89,
   #            'band_gap_ev': 2.35,
   #            'bulk_modulus_GPa': 95.3,
   #            ...}
"""

import threading
from typing import Dict, Optional
import torch
import numpy as np

_PES_MODEL = None           # 势能模型：能预测 E/forces/stress
_EFORM_MODEL = None         # 形成能模型
_BANDGAP_MODEL = None       # 带隙模型
_MODEL_LOCK = threading.Lock()


def _load_if_needed(model_name: str, cache_var: Optional[object]) -> Optional[object]:
    """单例加载 matgl 模型的通用辅助函数。"""
    if cache_var is not None:
        return cache_var
    try:
        import matgl
        return matgl.load_model(model_name)
    except Exception as e:
        print(f"[加载失败] {model_name}: {e}")
        return None


def predict_formation_energy(structure) -> Optional[float]:
    """eV/atom。优先使用 M3GNet 专用形成能模型。"""
    global _EFORM_MODEL
    with _MODEL_LOCK:
        _EFORM_MODEL = _load_if_needed("M3GNet-Eform-MP-2019.4.1", _EFORM_MODEL)
    if _EFORM_MODEL is None:
        return None
    try:
        return round(float(_EFORM_MODEL.predict_structure(structure)), 4)
    except Exception as e:
        print(f"[Eform 推理失败] {e}")
        return None


def predict_band_gap(structure) -> Optional[float]:
    """eV（PBE 级别）。"""
    global _BANDGAP_MODEL
    with _MODEL_LOCK:
        _BANDGAP_MODEL = _load_if_needed("MEGNet-MP-2019.4.1-BandGap-mfi", _BANDGAP_MODEL)
    if _BANDGAP_MODEL is None:
        return None
    try:
        return round(float(_BANDGAP_MODEL.predict_structure(structure)), 3)
    except Exception:
        return None


def predict_bulk_modulus(structure, n_sample_volumes: int = 7, vol_range: float = 0.04) -> Optional[float]:
    """从能量-体积曲线推导体模量（单位：GPa）。
    步骤：
    1. 在平衡体积 V0 的 ±4% 范围内，构造 n_sample_volumes 个缩放晶胞
    2. 每个缩放晶胞用 PES 模型预测总能量
    3. 用 Birch-Murnaghan 状态方程拟合 E(V)，拟合参数 B0 就是体模量
    """
    global _PES_MODEL
    with _MODEL_LOCK:
        _PES_MODEL = _load_if_needed("M3GNet-MP-2021.2.8-PES", _PES_MODEL)
    if _PES_MODEL is None:
        return None

    try:
        from pymatgen.analysis.eos import EOS

        V0 = structure.volume
        volumes, energies = [], []

        for scale in np.linspace(1 - vol_range, 1 + vol_range, n_sample_volumes):
            scaled = structure.copy()
            scaled.scale_lattice(V0 * scale)          # pymatgen 内置的晶格缩放
            E_total = float(_PES_MODEL.predict_structure(scaled))
            volumes.append(V0 * scale)
            energies.append(E_total)

        # Birch-Murnaghan 方程拟合：E(V) = E0 + (9B0/16)·[(V0/V)^(2/3)-1]^2·...
        eos = EOS(eos_name="birch_murnaghan")
        fit = eos.fit(volumes, energies)
        # fit.b0 是体模量（单位：eV/Å³），转换为 GPa（1 eV/Å³ = 160.218 GPa）
        B0_GPa = fit.b0 * 160.218
        return round(float(B0_GPa), 2)
    except Exception as e:
        print(f"[Bulk Modulus 推理失败] {e}")
        return None


def predict_all_properties(structure) -> Dict[str, Optional[float]]:
    """对一个 Structure 预测所有可预测属性的统一入口。"""
    return {
        "formation_energy_ev_per_atom": predict_formation_energy(structure),
        "band_gap_ev": predict_band_gap(structure),
        "bulk_modulus_GPa": predict_bulk_modulus(structure),
    }
```

#### 把这个模块接入 Flask 服务

在 `app.py` 顶部新增：

```python
from predict_properties import predict_all_properties
```

然后在 `/predict` 路由中，把原来的 `predict_formation_energy(s)` 替换为：

```python
props = predict_all_properties(s)
predictions.append({
    'formula': s.composition.reduced_formula,
    'key': key,
    'index': i + 1,
    **props,  # 直接展开所有属性（形成能、带隙、体模量...）
    'stability': '稳定' if (props['formation_energy_ev_per_atom'] is not None
                            and props['formation_energy_ev_per_atom'] < 0.0) else '不稳定',
    'lengths': info['lengths'],
    'angles': info['angles']
})
```

**这套方案的优点**：
- 结构清晰，每个属性预测独立——未来想加新属性（如剪切模量、热导率），只需新增 `predict_xxx()` 函数，不影响其他代码
- PES 模型做能量/应力评估比专用形成能模型更可靠（因为它能考虑应力响应，而不仅仅是单点能量）
- 体模量等力学属性的计算方法是物理上标准的（Birch-Murnaghan EOS），结果可用于科研论文

**需要注意的问题**：
1. **计算开销**：predict_bulk_modulus 对每个结构要做 7 次 PES 推理（7 个缩放后的晶胞），批量筛选时要考虑时间成本。可以在前端加个"仅 TOP-10 结构计算体模量"的开关
2. **PES 模型的单位问题**：不同 matgl 模型输出能量的单位可能不同（有的是 eV/晶胞，有的是 eV/atom），使用前要做 `float(model.predict_structure(s))` 并查文档确认单位
3. **负体积拟合失败**：极端结构可能导致 EOS 拟合发散，需要 try-except 包住

### 3.5 关于不确定性估计（让结果更"可信"）

无论用哪个方案，在给用户展示预测值时，**最好同时展示不确定度**，例如：

```
带隙：2.35 ± 0.15 eV
形成能：-0.89 ± 0.04 eV/atom
```

matgl 4.0 已经内置了 `MCDropoutWrapper`（蒙特卡洛 Dropout），可以用于不确定性估计。用法：

```python
from matgl.utils import MCDropoutWrapper

model = matgl.load_model("M3GNet-Eform-MP-2019.4.1")
wrapper = MCDropoutWrapper(model, n_passes=20)  # 用 20 次随机前向传递

mean, std = wrapper.predict_uncertainty([structure])
# mean: 20 次预测的平均值 → 作为最终预测值
# std: 20 次预测的标准差 → 作为模型的"不确定度"

result = f"{mean:.4f} ± {std:.4f}"
```

建议至少对 TOP-5 的结构做不确定性估计——这能让你的系统从"一个黑盒的 AI 工具"升级为"带有置信度的科学助手"。

---

## 第四部分：源码精读

### 4.1 关键文件清单

| 文件 | 行数 | 作用 | 需理解的关键概念 |
|-----|------|------|-----------------|
| `DiffCSP/diffcsp/pl_modules/diffusion.py` | ~240 | 扩散模型的训练正向传播和 PC 采样 | DDPM 公式、Predictor-Corrector、包裹正态 |
| `DiffCSP/diffcsp/pl_modules/cspnet.py` | ~290 | 解码器 CSPNet | 图神经网络、消息传递、正弦位置编码 |
| `DiffCSP/diffcsp/pl_modules/diff_utils.py` | ~110 | Beta/Sigma 调度器 + 包裹正态 pdf/score | 噪声 schedule |
| `DiffCSP/diffcsp/common/constants.py` | ~30 | 化学符号列表（H 到 Lr） | 元素索引映射 |
| `DiffCSP/diffcsp/common/data_utils.py` | 较多 | `lattice_params_to_matrix_torch` 等辅助函数 | 晶格参数 ↔ 基矢矩阵 |
| `DiffCSP/scripts/eval_utils.py` | ~335 | 模型加载 + `prop_model_eval()` + 结构有效性检查 | 生产环境的模型使用模式 |
| `diffcsp_integration.py` | ~180 | 你的项目对 DiffCSP++ 的封装 | 如何把学术代码转为 API |
| `app.py` | ~830 | Flask 服务 + M3GNet 集成 + 降级链 + 路由 | Web 服务设计模式 |

### 4.2 你最应该关心的几个函数/参数

**(1) `step_lr`（扩散采样的学习率）**

```python
# diffcsp_integration.py
outputs, _ = model.sample(batch, step_lr=1e-5)
```

这是 **PC 采样中校正器步的学习率**。在 `DiffCSP/scripts/eval_utils.py` 中也有推荐值表：

```python
# mp_csp: step_lr = 1e-5
# 太大 → 不稳定，生成不合理解构
# 太小 → 收敛慢，需要更多采样
```

如果你发现生成的结构"原子撞在一起"或"晶胞不合理"，首先调小这个值（比如 5e-6 或 1e-6）。

**(2) `num_structures`（每个化学式并行生成几个结构）**

```python
generate_structures_diffcsp(formula, num_structures=5)
```

DiffCSP++ 是随机的——每次生成都会得到略有不同的结构。`num_structures=5-10` 可以让你看到结构多样性，然后用 M3GNet 选能量最低的那个。

**(3) M3GNet 模型的缓存**

```python
# app.py
_MODEL = None
_MODEL_LOCK = threading.Lock()
```

这是**标准的线程安全单例模式**。Flask 默认在多线程模式下跑，不加锁可能导致两个请求同时触发模型加载——模型文件 100-500MB，重复加载会 OOM（内存溢出）。

**(4) 稳定性阈值 `eform < 0.0`**

```python
'stability': '稳定' if (eform is not None and eform < 0.0) else '不稳定'
```

这是**热力学意义上的严格判定**。形成能为负意味着"单质 → 化合物的反应放热"，即化合物是热力学稳定的。

### 4.3 常见坑与调试技巧

**坑 1：DiffCSP++ 生成的结构原子距离过近**
- 可能原因：`step_lr` 太大，或 T=1000 步不够收敛
- 调试：`eval_utils.py` 中有 `structure_validity(structure, cutoff=0.5)`，可以用它过滤输出。在 `diffcsp_integration.py` 的结尾加个过滤：
  ```python
  if min_distance > 0.5 and volume > 0.1:  # 保留合理结构
      results.append((s, info))
  ```

**坑 2：M3GNet 对某个结构报错**
- 可能原因：结构太离奇，比如所有原子都挤在一个点
- 调试：在 `predict_formation_energy()` 中已经有 try-except，失败时返回 None 即可。你可以在前端用"⚠️ 无法评估"这种图标展示

**坑 3：matgl 模型下载失败/联网失败**
- 原因：matgl 首次运行会从 HuggingFace 下载权重（几百 MB），需要联网
- 解决方案：在有网的环境先跑一次 `matgl.load_model(...)` 让它缓存到 `~/.cache/matgl/`，之后离线也能用；或手动拷贝缓存目录

**坑 4：CIF 文件用 VESTA 打开看起来很奇怪**
- 可能原因：pymatgen Structure 的分数坐标已经是 [0,1)，但某些 CIF 阅读器要求 Cartesian 坐标；或者对称性分析需要做 P1 变换
- 调试：在写入 CIF 前做 `structure.get_sorted_structure()` 或使用 `pymatgen.symmetry.analyzer.SpacegroupAnalyzer` 获得对称化后的原胞

### 4.4 性能优化建议

| 操作 | 预期效果 | 实现难度 |
|-----|---------|---------|
| 把 `num_structures` 从 5 提到 20-30（批量生成模式） | 获得更多结构候选，M3GNet 能挑出更稳的 | 低 |
| 预先加载所有模型到内存（启动 Flask 时调用一次 `get_model()` + `get_bandgap_model()`） | 首次请求响应时间从 10-30 秒降到 1-3 秒 | 极低 |
| 使用 CUDA（GPU）运行 DiffCSP++ 和 M3GNet | 采样速度提升 5-20 倍 | 低（只要改 `.cuda()` 或 `.to(device)`） |
| 对 TOP-N 结构做"快速 M3GNet 弛豫"（用 `Relaxer`） | 形成能精度显著提升，尤其是几何不合理的结构 | 中（需要调用 ASE 做结构优化） |
| 把形成能 + 带隙 + 体模量 合并为单次请求 | 减少前端-后端往返 | 低 |

---

## 第五部分：可执行的最小 Demo（学习 DiffCSP++ 原理用）

如果你想在不启动整个 Flask 服务的情况下，跑通 DiffCSP++ + M3GNet 的最小链路，用这段代码：

```python
"""
最小 Demo：DiffCSP++ 生成 AlPS3I → M3GNet 打分 → 打印结果
运行： python demo_diffcsp_m3gnet.py
"""
import sys
sys.path.insert(0, r"G:\TRAE\jiegouyuce")

from diffcsp_integration import generate_structures_diffcsp
from predict_properties import predict_all_properties   # 如果你已经写了这个模块

# 1. 生成 3 个结构
print("=== Step 1: DiffCSP++ 生成 3 个 AlPS3I 结构 ===")
structures = generate_structures_diffcsp("AlPS3I", num_structures=3)
print(f"成功生成 {len(structures)} 个结构\n")

# 2. 对每个结构做属性预测
print("=== Step 2: M3GNet 属性预测 ===")
for i, (s, info) in enumerate(structures):
    print(f"\n结构 #{i+1}: V = {s.volume:.1f} Å³, "
          f"a={info['lengths'][0]:.3f}, c={info['lengths'][2]:.3f}")

    # 形成能
    props = predict_all_properties(s)
    for key, val in props.items():
        print(f"  {key}: {val}")

# 3. 按形成能排序
print("\n=== Step 3: 按形成能排序 ===")
ranked = sorted(structures, key=lambda item: predict_all_properties(item[0])['formation_energy_ev_per_atom'] or float('inf'))
for rank, (s, info) in enumerate(ranked):
    e = predict_all_properties(s)['formation_energy_ev_per_atom']
    print(f"#{rank+1}: Eform = {e:+.4f} eV/atom  V = {s.volume:.1f} Å³")

# 4. 保存 CIF
for i, (s, _) in enumerate(ranked):
    from pymatgen.io.cif import CifWriter
    CifWriter(s).write_file(f"AlPS3I_rank{i+1}.cif")
print("\n已保存 CIF 文件")
```

跑这段代码能让你验证：
- DiffCSP++ 模型能否正确加载并生成合理结构
- M3GNet / matgl 模型能否正确加载并输出能量
- 从化学式到 CIF 文件的完整链路是否通顺

---

## 第六部分：总结与建议

### 你现在这套系统已经做到的

| 能力 | 状态 |
|-----|------|
| 基于化学式生成晶体结构（DiffCSP++） | ✅ 已实现 |
| 形成能评估（M3GNet） | ✅ 已实现 |
| 降级链（模型加载失败也不崩溃） | ✅ 已实现 |
| Three.js 3D 可视化 | ✅ 已实现 |
| CIF 文件导出 | ✅ 已实现 |
| 批量高通量生成 | ✅ 已有 `batch_generate_v2.py` |

### 建议的下一步（按优先级排）

| # | 方向 | 工作量估计 | 对你项目的价值 |
|---|------|-----------|-------------|
| 1 | **把 `predict_properties.py` 多属性预测模块写好**（形成能 + 带隙 + 体模量） | 2-3 小时编码 | ⭐⭐⭐⭐⭐ 立即提升系统功能 |
| 2 | **对 TOP-N 结构做 M3GNet 结构弛豫**（matgl 有 `Relaxer` 工具，用 `M3GNet-MP-2021.2.8-PES` 做结构优化后再评分） | 半天-1 天 | ⭐⭐⭐⭐⭐ 提升能量评估精度，更接近 DFT |
| 3 | **不确定性估计**：给每个预测值加 ± 区间 | 1 小时 | ⭐⭐⭐⭐ 让结果更具科学可信度 |
| 4 | **结构多样性分析**：TOP-N 结构两两做 RMSD，确认"最佳"不是运气 | 半天 | ⭐⭐⭐ 科研写作时有用 |
| 5 | **DiffCSP++ 属性预测头**：在生成端内部打分 | 1-2 周训练 | ⭐⭐ 可选研究方向 |
| 6 | **Docker 化部署** | 1-2 天 | ⭐⭐ 如果想分享给其他研究者使用 |

### 给你写简历/答辩材料时的建议

不要写成"我用 DiffCSP++ 和 M3GNet 做了一个网页"——这听起来像玩具项目。应该写成：

> **"基于条件扩散模型和三体图神经网络的硫磷卤化物高通量结构预测系统"**
> - 集成 DiffCSP++ 扩散模型实现化学式条件结构生成，支持 72 种候选化学式批量生成
> - 用 M3GNet 形成能模型作为能量评估器，替换原有的启发式元素焓值方法，实现了同一化学式下不同结构的区分能力
> - 构建基于 Flask 的交互式 Web 服务，集成 Three.js 3D 可视化，提供 CIF 文件导出用于后续 DFT 验证
> - 设计三级模型降级链 + 线程安全单例模型缓存，保证离线 / 受限网络环境下系统仍可运行
> - 体系扩展性：预留带隙、体模量等属性预测接口，可通过 matgl 预训练模型快速扩展

祝顺利！
