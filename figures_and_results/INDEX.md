# figures_and_results — 目录说明

论文所有绘图脚本、处理好的数据、生成图与输入素材的归档。

## 目录结构

- `scripts/` — 主文图的绘制脚本与共享模块（`config.py` / `libformat.py`）
- `data/` — 绘图用的处理后数据文件（无扩展名，纯文本数值表）
- `figures/` — 主文生成图（scaling / benchmarking / isol24 / locality / solid / unbiaseness）
- `assets/` — 绘图输入的图片素材（如 H-hBN 超胞示意图 `hBN_3x5.png`）
- `diagrams/` — 方法示意图（tensor contraction / (T) 等），由 `plot_diagram*.py` 生成 `diagram_*.png`
- `results_benchmarking/` — 分子基准（对比 DLPNO / exact）的数据与 MAE 计算脚本
- `SI/` — 补充材料图（收敛性、canonical/Haar 缩放）的脚本与数据

## scripts/ 说明

| 文件 | 作用 | 读取数据 | 输出图 |
|------|------|----------|--------|
| `config.py` | 颜色/字体配置 + 路径常量 `DATA_DIR`/`FIG_DIR`/`ASSET_DIR` | — | — |
| `libformat.py` | 对数刻度等绘图辅助函数 | — | — |
| `main_plot_scaling.py` | 图：水团簇 `N_sample` 与耗时随体系规模缩放 | `Nsample_water`, `stats_water` | `scaling.png` |
| `main_plot_benchmarking.py` | 分子基准：误差/耗时对比 DLPNO & exact | `stats_all` | `benchmarking.png` |
| `main_plot_isol24.py` | ISOL24 反应集基准 | `stats_isol24` | `isol24.png`, `isol24.pdf` |
| `main_plot_locality.py` | 局域轨道性能（PAH / H-hBN） | `data_locality` (+ `hBN_3x5.png`) | `locality.png` |
| `main_plot_solid.py` | 周期性 Si-掺杂金刚石耗时缩放 | （硬编码数值） | `solid.png` |
| `main_plot_unbiaseness.py` | 苯的无偏性验证（CCSD / (T) 误差分布） | `benzene_energy_CCSD_tz`, `benzene_energy_pt_tz` | `unbiaseness.png` |

**运行约定**：脚本内的数据/图片/素材路径通过 `config.data()` / `config.fig()` / `config.asset()`
解析为相对于脚本位置的绝对路径，因此无论从哪个工作目录运行都能正确找到文件。

## data/ 数据文件

| 文件 | 用途 |
|------|------|
| `Nsample_water` | 水团簇达到目标误差所需样本数 |
| `stats_water` | 水团簇各实现的计算耗时 |
| `stats_all` | 分子基准集合的误差/耗时汇总 |
| `stats_isol24` | ISOL24 反应集的误差/耗时 |
| `data_locality` | PAH / H-hBN 不同超胞的 `N_sample` 与耗时 |
| `benzene_energy_CCSD_tz` | 苯 cc-pVTZ 下 STC-CCSD 能量样本 |
| `benzene_energy_pt_tz` | 苯 cc-pVTZ 下 STC-(T) 能量样本 |
