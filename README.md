# LLMRF 项目使用说明

## ✅ 已完成修复

1. **环境变量设置**：在导入模块之前设置 `HPEESOF_DIR`
2. **导入顺序修复**：`keysight.ads.dds` 必须在 `keysight.ads.dataset` 之前导入
3. **代码语法错误**：修复了 visualizer.py 的函数定义问题
4. **启动脚本**：创建 `run.py` 使用正确的 ADS Python 解释器

## 🚀 使用方法

### 方式一：使用启动脚本（推荐）

在项目根目录（`d:\Desktop\sjtu\LLMRF`）运行：

```bash
python run.py --workspace <工作空间路径> --library <库名> --design <设计名>

# 示例
python run.py --workspace D:\Desktop\test_wrk --library test_lib --design test
```

### 方式二：直接使用 ADS Python

```bash
cd adsapi
"F:/Program Files (x86)/ADS2026/tools/python/python.exe" main.py --workspace <路径> --library <库名> --design <设计名>
```

## 🔧 配置文件

只需要修改 `adsapi/config.json`：

```json
{
  "ads": {
    "install_path": "F:/Program Files (x86)/ADS2026",
    "python_exe": "F:/Program Files (x86)/ADS2026/tools/python/python.exe"
  },
  "project": {
    "name": "LLMRF",
    "version": "2.0",
    "description": "ADS 自动化仿真系统"
  },
  "output": {
    "results_dir": "./test_plots",
    "formats": ["png", "pdf"]
  }
}
```

## 📝 常见问题

### Q: 出现 "No module named 'keysight'" 错误
**A**: 必须使用 ADS 自带的 Python 解释器，路径在 `F:/Program Files (x86)/ADS2026/tools/python/python.exe`

### Q: 出现 "HPEESOF_DIR must be set" 错误
**A**: 已修复，环境变量会在导入模块之前设置

### Q: Qt 字体警告
**A**: 可以忽略，这些是 ADS Python 的 Qt 库的字体加载警告，不影响功能

### Q: 导入顺序错误
**A**: 已修复，`dds` 会在 `dataset` 之前导入

## 📁 项目结构

```
LLMRF/
├── run.py                    # 启动脚本（项目根目录）
└── adsapi/
    ├── config.json          # 配置文件（唯一需要修改）
    ├── main.py             # 主程序
    ├── ads_engine.py       # ADS 仿真引擎
    ├── netlist_parser.py   # 网表解析
    ├── visualizer.py       # 结果可视化
    ├── llm_interface.py    # LLM 接口
    ├── check_config.py     # 配置检查
    ├── test_imports.py     # 导入测试
    └── 项目文档.md         # 详细文档
```

## 🧪 测试

### 检查配置
```bash
cd adsapi
python check_config.py
```

### 测试模块导入
```bash
cd adsapi
"F:/Program Files (x86)/ADS2026/tools/python/python.exe" test_imports.py
```

### 运行仿真
```bash
cd ..  # 回到项目根目录
python run.py --workspace D:\Desktop\test_wrk --library test_lib --design test
```

## 📊 核心功能

1. **无 GUI 运行**：完全命令行，不需要打开 ADS GUI
2. **自动仿真**：读取网表 → 运行仿真 → 提取数据
3. **结果可视化**：自动生成 S 参数图表、保存为 PNG/PDF
4. **数据提取**：使用 `dataset.open()` 读取 .ds 文件转为 DataFrame
5. **单一配置**：所有设置都在 config.json 中

## ⚠️ 注意事项

1. **必须使用 ADS Python**：系统 Python 无法导入 keysight 模块
2. **导入顺序**：代码中已确保正确的导入顺序
3. **环境变量**：`HPEESOF_DIR` 会自动设置
4. **工作空间路径**：使用绝对路径

## 📖 详细文档

更多详细信息请查看 `adsapi/项目文档.md`
