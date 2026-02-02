"""
LLMRF 项目启动脚本
使用 ADS Python 环境运行主程序
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def main():
    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()
    config_path = project_root / 'adsapi' / 'config.json'
    
    # 读取配置
    if not config_path.exists():
        print("❌ 未找到配置文件：config.json")
        return 1
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取 ADS Python 解释器路径
    python_exe = config['ads']['python_exe']
    
    if not Path(python_exe).exists():
        print(f"❌ 未找到 ADS Python 解释器: {python_exe}")
        print("   请检查 config.json 中的 ads.python_exe 配置")
        return 1
    
    # 准备命令行参数
    main_script = project_root / 'adsapi' / 'main.py'
    
    # 传递所有命令行参数
    args = [python_exe, str(main_script)] + sys.argv[1:]
    
    print(f"✓ 使用 ADS Python: {python_exe}")
    print(f"✓ 运行脚本: {main_script}")
    print(f"✓ 参数: {' '.join(sys.argv[1:])}")
    print("-" * 60)
    
    # 运行主程序
    try:
        result = subprocess.run(args, cwd=str(project_root / 'adsapi'))
        return result.returncode
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
