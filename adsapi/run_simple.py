"""
LLMRF 极简启动器

直接用 Python 运行，无需 PowerShell 权限
适配 ADS Python 环境，无需额外配置

使用方法：
    "F:/Program Files (x86)/ADS2026/tools/python/python.exe" run_simple.py
"""

import os
import sys
import json
import subprocess


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        print("❌ 找不到 config.json 文件")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_ads_python(config):
    """检查 ADS Python 是否存在"""
    python_exe = config['ads']['python_exe']
    
    if not os.path.exists(python_exe):
        print(f"❌ 找不到 ADS Python: {python_exe}")
        print(f"   请在 config.json 中修改 ads.python_exe 路径")
        return False
    
    print(f"✅ ADS Python: {python_exe}")
    return True


def main():
    print("=" * 70)
    print("LLMRF 极简启动器")
    print("=" * 70)
    print()
    
    # 加载配置
    config = load_config()
    if not config:
        input("\n按 Enter 退出...")
        return 1
    
    # 检查 ADS Python
    if not check_ads_python(config):
        input("\n按 Enter 退出...")
        return 1
    
    # 设置环境变量
    ads_dir = config['ads']['install_path']
    os.environ['HPEESOF_DIR'] = ads_dir
    print(f"✅ HPEESOF_DIR: {ads_dir}")
    print()
    
    # 显示菜单
    print("请选择运行模式:")
    print("  1. 测试模块（推荐首次运行）")
    print("  2. 自动仿真（使用当前 ADS 工作空间）")
    print("  3. 指定工作空间仿真")
    print()
    
    try:
        choice = input("请输入选项 (1/2/3): ").strip()
    except KeyboardInterrupt:
        print("\n\n已取消")
        return 0
    
    python_exe = config['ads']['python_exe']
    main_py = os.path.join(os.path.dirname(__file__), 'main.py')
    
    if choice == '1':
        print("\n" + "=" * 70)
        print("运行测试模式...")
        print("=" * 70)
        print()
        
        # 测试各个模块
        test_files = ['netlist_parser.py', 'visualizer.py', 'llm_interface.py']
        
        for test_file in test_files:
            test_path = os.path.join(os.path.dirname(__file__), test_file)
            print(f"\n🧪 测试 {test_file}...")
            print("-" * 70)
            
            result = subprocess.run(
                [python_exe, test_path],
                env=os.environ.copy()
            )
            
            if result.returncode != 0:
                print(f"❌ {test_file} 测试失败")
            else:
                print(f"✅ {test_file} 测试通过")
    
    elif choice == '2':
        print("\n" + "=" * 70)
        print("自动仿真模式...")
        print("=" * 70)
        print()
        
        cmd = [python_exe, main_py, '--auto', '--output', config['output']['dir']]
        
        if config['llm']['enabled']:
            cmd.extend(['--use-llm', '--llm-url', config['llm']['api_url'], '--llm-model', config['llm']['model']])
        
        print(f"执行命令: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    
    elif choice == '3':
        print("\n" + "=" * 70)
        print("指定工作空间模式...")
        print("=" * 70)
        print()
        
        workspace = input("工作空间路径: ").strip()
        library = input("库名称: ").strip()
        design = input("设计名称: ").strip()
        
        if not all([workspace, library, design]):
            print("❌ 必须提供所有参数")
            input("\n按 Enter 退出...")
            return 1
        
        cmd = [
            python_exe, main_py,
            '--workspace', workspace,
            '--library', library,
            '--design', design,
            '--output', config['output']['dir']
        ]
        
        if config['llm']['enabled']:
            cmd.extend(['--use-llm', '--llm-url', config['llm']['api_url'], '--llm-model', config['llm']['model']])
        
        print(f"\n执行命令: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, env=os.environ.copy())
        return result.returncode
    
    else:
        print("❌ 无效的选项")
        input("\n按 Enter 退出...")
        return 1
    
    print("\n" + "=" * 70)
    print("执行完成")
    print("=" * 70)
    input("\n按 Enter 退出...")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 退出...")
        sys.exit(1)
