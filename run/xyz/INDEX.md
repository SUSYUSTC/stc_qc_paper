# run/xyz — 几何结构说明

计算所用的分子 / 周期体系几何结构（`.xyz`）与晶格（`.lattice`）文件。

| 子目录 | 内容 |
|--------|------|
| `water/` | 水团簇几何（`waterN_min.xyz`），用于 STC-CCSD 缩放与无偏性研究 |
| `benchmarking/` | 分子基准集合（对比 DLPNO / exact）的几何 |
| `ISOL24/` | ISOL24 反应集的 22 个体系几何 |
| `cell/` | 周期性体系（如 Si-掺杂金刚石）的 `.xyz` 与对应 `.lattice` 晶格文件（成对存放） |
| `lattice/` | H-hBN 与 PAH 超胞几何（用于局域轨道与固体耗时图） |
| `benzene/` | 苯单分子几何 `benzene.xyz`（无偏性验证用） |

注：`.lattice` 文件与对应 `.xyz` 配套，配合驱动脚本的 `--pbc` 选项使用。
