# Docs / 深入阅读

All deep-dive notes are in Chinese (the project author's working language). The code and README are in English.

| Document | What's inside |
|---|---|
| [晶体结构预测项目_技术说明报告.md](晶体结构预测项目_技术说明报告.md) | Overall technical report of the research pipeline: DiffCSP++ generation → M3GNet scoring → TOP30 screening, with result tables |
| [CDVAE和DiffCSP++入门教程_面向小白.md](CDVAE和DiffCSP++入门教程_面向小白.md) | Beginner-friendly intro to diffusion-based crystal structure prediction (CDVAE vs DiffCSP++) |
| [DiffCSP++原理详解与扩展路线图.md](DiffCSP++原理详解与扩展路线图.md) | Math details of the diffusion model (denoising on fractional coords + lattice), and extension roadmap |
| [P1空间群CIF输出_技术说明文档.md](P1空间群CIF输出_技术说明文档.md) | Why DiffCSP++ outputs are P1-certified CIFs, and how spglib re-detects symmetry (symprec=0.1) |
