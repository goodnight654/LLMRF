"""
配置检查工具 - 快速诊断 ADS 路径配置问题

运行此脚本检查 config.json 中的路径是否正确

使用方法：
    python check_config.py
或
    "F:/Program Files (x86)/ADS2026/tools/python/python.exe" check_config.py
"""

import os
import json


def check_config():
    """检查配置文件"""
    print("=" * 70)
    print("LLMRF 配置检查工具")
    print("=" * 70)
    print()
    
    # 读取配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        print("❌ 找不到 config.json 文件")
        return False
    
    print("✅ 找到配置文件: config.json")
    print()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 检查 ADS 安装路径
    print("检查 ADS 配置...")
    print("-" * 70)
    
    install_path = config['ads']['install_path']
    python_exe = config['ads']['python_exe']
    
    print(f"install_path: {install_path}")
    if os.path.exists(install_path):
        print("  ✅ 路径存在")
    else:
        print("  ❌ 路径不存在！")
        print(f"     请检查是否正确安装 ADS")
        return False
    
    print()
    print(f"python_exe: {python_exe}")
    if os.path.exists(python_exe):
        print("  ✅ Python 解释器存在")
    else:
        print("  ❌ Python 解释器不存在！")
        print(f"     请检查路径是否正确")
        return False
    
    # 检查路径一致性
    print()
    print("检查路径一致性...")
    print("-" * 70)
    
    # Python 路径应该在 install_path 下
    python_normalized = os.path.normpath(python_exe)
    install_normalized = os.path.normpath(install_path)
    
    if python_normalized.startswith(install_normalized):
        print("✅ Python 路径与安装路径一致")
    else:
        print("⚠ 警告：Python 路径与安装路径不一致")
        print(f"   install_path: {install_normalized}")
        print(f"   python_exe:   {python_normalized}")
        print()
        print("建议修正：")
        # 从 python_exe 推断正确的 install_path
        # python.exe 通常在 <install_path>/tools/python/python.exe
        if 'tools\\python\\python.exe' in python_normalized or 'tools/python/python.exe' in python_normalized:
            suggested_install = python_normalized.split('tools')[0].rstrip('\\/').replace('\\', '/')
            print(f'   "install_path": "{suggested_install}"')
    
    print()
    print("检查关键目录...")
    print("-" * 70)
    
    # 检查关键子目录
    key_dirs = [
        'bin',
        'tools/python',
        'tools/python/packages/keysight'
    ]
    
    all_ok = True
    for subdir in key_dirs:
        full_path = os.path.join(install_path, subdir)
        if os.path.exists(full_path):
            print(f"✅ {subdir}")
        else:
            print(f"❌ {subdir} (不存在)")
            all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✅ 配置检查通过！")
        print()
        print("建议的 HPEESOF_DIR 环境变量：")
        print(f"  {install_path}")
        return True
    else:
        print("❌ 配置存在问题，请修正 config.json")
        print()
        print("常见问题：")
        print("1. 路径使用正斜杠 / 而不是反斜杠 \\")
        print("2. install_path 应该指向 ADS 安装根目录")
        print("3. python_exe 应该是 <install_path>/tools/python/python.exe")
        return False


def suggest_config():
    """根据常见安装位置建议配置"""
    print()
    print("=" * 70)
    print("自动检测 ADS 安装...")
    print("=" * 70)
    print()
    
    # 常见安装位置
    common_paths = [
        "C:/Program Files/Keysight/ADS2026",
        "C:/Program Files (x86)/Keysight/ADS2026",
        "F:/Program Files/Keysight/ADS2026",
        "F:/Program Files (x86)/Keysight/ADS2026",
        "C:/ADS2026",
        "D:/ADS2026",
        "F:/ADS2026",
        "C:/Program Files/ADS2026",
        "C:/Program Files (x86)/ADS2026",
        "F:/Program Files/ADS2026",
        "F:/Program Files (x86)/ADS2026",
    ]
    
    found = []
    for path in common_paths:
        python_exe = os.path.join(path, "tools/python/python.exe")
        if os.path.exists(python_exe):
            found.append(path)
            print(f"✅ 找到 ADS: {path}")
    
    if found:
        print()
        print("建议配置（复制到 config.json）：")
        print("-" * 70)
        best = found[0]
        print(f'''{{
  "ads": {{
    "install_path": "{best.replace(chr(92), '/')}",
    "python_exe": "{best.replace(chr(92), '/')}/tools/python/python.exe"
  }}
}}''')
    else:
        print("❌ 未找到 ADS 安装")
        print()
        print("请手动检查 ADS 安装位置，然后修改 config.json")


if __name__ == "__main__":
    success = check_config()
    
    if not success:
        suggest_config()
    
    print()
    input("按 Enter 键退出...")
