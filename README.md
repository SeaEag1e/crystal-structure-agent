---
title: Crystal Structure Agent
emoji: 💎
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: true
---

# 💎 Crystal Structure Agent

我在做层状磷酸卤化物（MPX₃I 族，比如 GaPSe₃I）的计算筛选。每筛一个化学式都要串五六个脚本：扩散模型生成候选结构 → 机器学习势打分 → 对称性分析 → 跟实验结构对比。一个组成盯下来半天就没了，脚本换个环境还经常跑不起来。

这个项目就是想把这套流程变成一个能对话的系统。输入"找出最稳定的结构并分析它们的对称性"，Agent 自己规划工具链、执行、返回一份带 3D 结构的报告，浏览器里直接旋转查看。

## 我主要想解决的问题

**科研流程自动化**：以前一个化学式要手工跑半天，现在一句话。Agent 会把复合需求拆成工具链，比如上面那句话会自动变成"查库 → 稳定性筛选 → 对称分析"三步执行。

**没有大模型也能用**：每个用到 LLM 的地方（意图识别、任务规划、结果汇总）都写了规则兜底。不填 API Key 系统照样完整跑通，只是从"大模型理解"退回到"关键词匹配"。这也是我对"怎么把科研流程放到 LLM Agent 后面还不脆弱"这个问题的回答：LLM 挂了系统不能挂。

**没有 GPU 也能演示**：扩散模型生成需要 GPU，云端免费环境跑不了。我的做法是在本地 GPU 上把 186 个结构全部预先生成好，打包成结构库上线。云端负责轻量的实时打分（M3GNet 只有 2.3MB）。GPU 生成工具在调用时会检查 CUDA，没有就返回"为什么做不了"的说明，而不是报错崩溃。

**全程可追溯**：Agent 每一步做了什么——识别出什么意图、生成了什么计划、调了哪个工具、传了什么参数——都折叠显示在对话页里，点开就能看。科研场景里这个比"直接给答案"重要。

## 能做什么

| 页面 | 功能 |
|---|---|
| 🏠 首页 | 项目简介、架构图、环境自检（M3GNet / API Key / CUDA 三个指示灯） |
| 💬 Agent 对话 | 自然语言查询，展示完整的意图 → 计划 → 工具轨迹 → 报告 |
| 🧊 结构库 | 按化学式/能量/空间群筛选 186 个结构，3D 球棍模型旋转查看，任意结构实时重新打分 |
| ⚡ 本地 GPU 模式 | 完整的 DiffCSP++ 扩散生成（1000 步采样），需要本地 CUDA 显卡 |

可以用中文或英文问，举几个例子：

```text
浏览 GaPS3I 的结构
找出最稳定的结构并分析它们的对称性        # 触发 3 步工具链：查库 → 筛选 → 对称分析
生成的结构和实验结构对比一下              # 触发与实验 C2/c 参照结构的对比
```

## 快速开始

```bash
git clone https://github.com/SeaEag1e/crystal-structure-agent.git
cd crystal-structure-agent
pip install -r requirements.txt
streamlit run app.py
```

CPU 就能跑。想启用大模型大脑的话，去[智谱开放平台](https://open.bigmodel.cn/)免费领一个 GLM-4-Flash 的 Key：

```bash
# Windows
set ZHIPU_API_KEY=your_key
# Linux/macOS
export ZHIPU_API_KEY=your_key
```

不填也行，规则模式功能完整。

> ⚠️ 别把 API Key 提交进 git。部署到线上时放在平台的 Secrets 配置里。

## 在线体验

仓库顶部带了 Streamlit Space 的配置头（就是本文件最上面那段 frontmatter），所以部署到 Hugging Face Spaces 只需要：

```bash
# 在 huggingface.co 上建一个 Space 后
git remote add space https://huggingface.co/spaces/SeaEag1e/crystal-structure-agent
git push space main
```

然后在 Space 的 Settings → Variables and secrets 里配上 `ZHIPU_API_KEY`。首次构建约 10 分钟（CPU 版 torch 比较大）。

## 技术上怎么实现的

```
┌────────────────────────────────────────────┐
│  Streamlit UI                              │
│  首页自检 │ Agent对话 │ 结构库 │ GPU生成   │
├────────────────────────────────────────────┤
│  LangGraph Agent                           │
│  意图识别 → 任务规划 → 工具调度(循环)      │
│                       ↘ 结果汇总           │
├────────────────────────────────────────────┤
│  6 个工具（白名单）                         │
│  查库 │ M3GNet打分 │ 对称分析 │ 稳定性筛选  │
│  实验对比 │ GPU生成                        │
├────────────────────────────────────────────┤
│  core/ 纯函数科学计算库                     │
│  structures │ scoring │ symmetry │ generation │
├────────────────────────────────────────────┤
│  预计算知识库                               │
│  index.csv (186结构) │ CIF │ M3GNet权重    │
└────────────────────────────────────────────┘
```

几个关键设计：

- **每个 LLM 边界都有规则兜底**。意图识别失败走关键词匹配，规划失败走组合链式规划，汇总失败走模板输出。断网、Key 错、JSON 返回乱码，系统都照常工作。
- **预计算 + 实时混合**。生成和批量打分在本地离线做完存进 index.csv，在线只做单结构实时打分。免费 CPU 也能秒级响应。
- **Agent 能感知硬件边界**。GPU 工具调用时检查 CUDA，不可用时返回降级说明，Agent 会在报告里告诉你为什么做不了、去哪里能做。
- **为免费资源做的性能控制**。模型和 LLM 客户端用 `st.cache_resource` 单例缓存，推理用 `torch.inference_mode()`，线程数按 2-vCPU 限制。

## 项目结构

```
crystal-structure-agent/
├── app.py                     # 首页：简介 + 架构 + 环境自检
├── pages/
│   ├── 1_Agent_Chat.py        # Agent 对话 + 轨迹展示
│   ├── 2_Structure_Library.py # 结构库 + 3D 查看器 + 重新打分
│   └── 3_Local_GPU_Mode.py    # DiffCSP++ 生成（需要 CUDA）
├── agent/                     # LangGraph 智能体
│   ├── graph.py               # 意图 → 规划 → 执行 → 汇总
│   ├── tools.py               # 6 个 @tool（白名单）
│   ├── prompts.py             # 四组提示词
│   ├── llm.py                 # GLM-4-Flash 接入
│   └── state.py               # AgentState 定义
├── core/                      # 纯函数科学计算库
│   ├── structures.py          # CIF 读写
│   ├── scoring.py             # M3GNet 形成能（含降级链）
│   ├── symmetry.py            # spglib 空间群 + 极性
│   └── generation.py          # DiffCSP++ 封装（延迟导入）
├── data/
│   ├── structures/            # 186 个生成的 CIF
│   ├── index.csv              # 预计算知识库
│   ├── reference/             # 实验 GaPSe3I 参照结构
│   └── models/matgl/          # 内嵌 M3GNet 权重（~2.3MB）
├── scripts/
│   ├── export_library.py      # 一站式数据管线（本地跑）
│   ├── generate_gpu.py        # GPU 生成命令行
│   └── smoke_test_agent.py    # Agent 端到端测试
└── docs/                      # 深入文档
```

## 测试

```bash
python scripts/smoke_test_agent.py
```

覆盖单意图浏览、复合意图多工具链（查库 → 筛选 → 对称分析、查库 → 实验对比）、计划白名单校验、无 GPU 环境的降级行为。当前规则模式和 LLM 模式全部通过，0 失败。

## 后续计划

现在这个版本还是"你问我答"，Agent 只能调度我写好的工具。下一步想让它更主动：

1. **DFT 二审**：TOP30 候选送 VASP/CP2K 弛豫，机器学习势初审 + DFT 终审，误报率会降不少
2. **多 Agent 互驳**：加一个"批评家"Agent，专门挑规划 Agent 工具选择的毛病，辩论后再执行
3. **XRD 多模态输入**：拍一张 XRD 谱图传上去，Agent 自己比对候选结构出匹配报告
4. **异步并行打分**：现在 M3GNet 打分是串行的，换 asyncio 能快几倍

## 致谢

这个项目站在两个开源模型的工作之上，向作者致谢：

- **DiffCSP++** —— Rui Jiao, Wenbing Huang, Yu Liu, Deli Zhao, Yang Liu（清华大学 & 人民大学），*Space Group Constrained Crystal Generation*, ICLR 2024。项目里的晶体结构就是用它生成的。代码：[jiaor17/DiffCSP-PP](https://github.com/jiaor17/DiffCSP-PP)
- **M3GNet** —— Chi Chen, Shyue Ping Ong（UCSD），*A universal graph deep learning interatomic potential for the periodic table*, Nature Computational Science 2022。项目里的形成能打分全靠它。代码：[materialsvirtuallab/matgl](https://github.com/materialsvirtuallab/matgl)

同时感谢 [spglib](https://github.com/spglib/spglib)（空间群分析）、[pymatgen](https://github.com/materialsproject/pymatgen)（晶体 IO）、智谱 AI 的 [GLM-4-Flash](https://open.bigmodel.cn/)（免费 LLM 推理）。

## 许可

MIT，见 [LICENSE](LICENSE)。M3GNet 权重按 matgl 项目条款再分发；DiffCSP++ 本体不在仓库里（GPU 生成需要自备 checkout）。
