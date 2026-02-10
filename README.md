# LLMRF 项目使用说明

本项目是一个 **LLM + Keysight ADS** 的射频电路自动化设计闭环原型：
- LLM（本地 Ollama）：用于参数建议/优化策略
- ADS Python API：用于自动生成电路、运行仿真、提取指标与可视化

## 项目进度与下一步

请先阅读根目录文档：
- `IMPLEMENTATION_STATUS.md`（进度总结 + 接下来每一步怎么做）

## 快速开始（推荐入口）

### 1) 先配置一次（唯一需要改的配置）

编辑 `adsapi/config.json`：
- `ads.install_path`：ADS 安装目录（例如 `C:/Program Files/Keysight/ADS2025_Update1`）
- `ads.python_exe`：ADS 自带 Python（例如 `C:/Program Files/Keysight/ADS2025_Update1/tools/python/python.exe`）

> 说明：文档/脚本里的 ADS 路径以 `adsapi/config.json` 为准，避免版本混用。

### 2) 运行仿真（根目录入口）

在项目根目录运行：

```bash
python run.py --workspace <工作空间路径> --library <库名> --design <设计名>

# 示例
python run.py --workspace D:\Desktop\test_wrk --library test_lib --design test
```

## 目录结构（核心）

```
LLMRF/
├── IMPLEMENTATION_STATUS.md  # 进度总结 + 下一步
├── run.py                    # 入口：用 ADS Python 跑 adsapi/main.py
├── filter_design.py          # demo：自动生成滤波器 + 仿真
└── adsapi/
    ├── config.json           # 唯一配置入口
    ├── main.py               # ADS 自动化主程序
    ├── ads_engine.py         # 仿真引擎封装
    ├── netlist_parser.py     # 网表变量解析
    ├── visualizer.py         # 绘图
    ├── post_processor.py     # HB 后处理（PAE/Gain/...）
    ├── batch_filter_simulation.py  # 批量滤波器仿真/数据集生成
    └── 项目文档.md           # adsapi 子系统详细文档
```

## 常见问题

### Q: 出现 "No module named 'keysight'" 错误
**A**: 必须使用 ADS 自带的 Python 解释器；`run.py` 会从 `adsapi/config.json` 读取并使用该解释器。

### Q: 出现 "HPEESOF_DIR must be set" 错误
**A**: `adsapi/main.py` 会尝试从 `adsapi/config.json` 设置 `HPEESOF_DIR`，请先确认配置里的 `ads.install_path` 正确。

### Q: Qt 字体警告
**A**: 一般可忽略，不影响仿真流程。

## 详细文档

- `adsapi/项目文档.md`：ADS 自动化子系统说明
- `PROJECT_GUIDE.md`：更完整的阶段指南与规划
