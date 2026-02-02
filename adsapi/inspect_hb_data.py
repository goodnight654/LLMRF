"""
查看 HB 数据的实际结构
"""
import os
import sys
import pandas as pd

# 设置环境变量
os.environ['HPEESOF_DIR'] = r"F:/Program Files (x86)/ADS2026"

from keysight.ads import dds
from keysight.ads import dataset
from pathlib import Path

# 打开数据集
ds_path = r"D:\Desktop\test_wrk\sim_output_1\test.ds"
output_data = dataset.open(Path(ds_path))

print("数据块列表:", output_data.varblock_names)

for block_name in output_data.varblock_names:
    print(f"\n{'='*70}")
    print(f"数据块: {block_name}")
    print('='*70)
    
    df = output_data[block_name].to_dataframe().reset_index()
    
    print(f"\n列名 ({len(df.columns)} 列):")
    for col in df.columns:
        print(f"  - {col}")
    
    print(f"\n前3行数据:")
    print(df.head(3))
    
    # 查找谐波数据
    print(f"\n查找可能的谐波数据模式:")
    harmonic_cols = [col for col in df.columns if '[' in col and ']' in col]
    if harmonic_cols:
        print(f"  找到 {len(harmonic_cols)} 个谐波相关列:")
        for col in harmonic_cols[:10]:  # 只显示前10个
            print(f"    - {col}")
