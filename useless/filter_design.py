import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.core import getipython
 
# 配置 ADS 环境
if "HPEESOF_DIR" not in os.environ:
    # 尝试自动设置，这里假设是之前配置中的路径
    os.environ["HPEESOF_DIR"] = r"C:\Program Files\Keysight\ADS2025_Update1"

hpeesof_dir = os.environ["HPEESOF_DIR"]
bin_path = os.path.join(hpeesof_dir, "bin")

if bin_path not in os.environ["PATH"]:
    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
 
from keysight.ads import de
from keysight.ads.de import db_uu as db

try:
    from keysight.edatoolbox import ads
except ImportError:
    # 尝试从 ADS 安装目录加载
    print("Warning: keysight.edatoolbox not found, attempting to load from ADS directory...")
    ads_site_packages = r"C:\Program Files\Keysight\ADS2025_Update1\tools\python\Lib\site-packages"
    if os.path.exists(ads_site_packages) and ads_site_packages not in sys.path:
        sys.path.append(ads_site_packages)
    from keysight.edatoolbox import ads

import keysight.ads.dataset as dataset
 
 
ripple_db = 0.1  # 通带波纹
fc = 800e6  # 通带截止频率(Hz)
fs = 1500e6  # 阻带频率(Hz)
R0 = 50  # 滤波器的参考阻抗
La = 40  # fs频率处所需的衰减(dB)
 
workspace_path = r"G:\wenlong\ADS\test_Filter"
cell_name = "test"
library_name = "test_lib"
 
 
# Chebyshev低通滤波器设计器
def lpf_design_by_Atten(ripple_db, fc, fs, R0, La):
    ep = 10 ** (ripple_db / 10) - 1
    pi = np.pi
    wc = 2 * pi * fc  # 角频通带频率
    ws = 2 * pi * fs  # 角频阻带频率
 
    N = round(np.sqrt(np.arccosh((10 ** (La) - 1) / ep)) / np.arccosh(ws / wc)) - 1
    # N公式
    # N = np.ceil( np.arccosh( np.sqrt( (10**(La/10) - 1) / ep ) ) / np.arccosh(ws / wc) )
    Atten = 0 - 10 * np.log10(1 + ep * np.cosh((N * np.arccosh(ws / wc))) ** 2)
    Atten = round(Atten, 2)
 
    beta = np.log(1 / np.tanh(ripple_db / 17.37))
    gamma = np.sinh(beta / (2 * N))
 
    L = []
    C = []
    ak = []
    bk = []
    gk = []
 
    for k in range(1, N + 1):
        a1 = np.sin(((2 * k - 1) * pi) / (2 * N))
        ak.append(a1)
 
        b1 = gamma**2 + (np.sin(k * pi / N)) ** 2
        bk.append(b1)
 
    for k in range(1, N + 1):
        if k == 1:
            gk.append(round(2 * ak[k - 1] / gamma, 4))
        else:
            gk.append(round((4 * ak[k - 2] * ak[k - 1]) / (bk[k - 2] * gk[k - 2]), 4))
 
        if k % 2 != 0:
            L.append(round(((R0 * gk[k - 1] / wc) / 1e-9), 2))
        else:
            C.append(round((gk[k - 1] / (R0 * wc)) / 1e-12, 2))
    return L, C, N, Atten, gk
 
 
# 使用所需衰减方法设计低通滤波器
L, C, N, La, gk = lpf_design_by_Atten(ripple_db, fc, fs, R0, La)
 
print("\ng值=", gk)
print("使用所需衰减方法的滤波器设计")
print("计算的滤波器阶数 =", N)
print("@fs处计算的衰减(dB) =", La)
print("L(nH) =", L)
print("C(pF) =", C)
 
 
def create_and_open_an_empty_workspace(workspace_path: str):
    # 确保没有已打开的工作空间
    if de.workspace_is_open():
        de.close_workspace()
 
    # 如果目录已存在，则无法创建工作空间
    if os.path.exists(workspace_path):
        print(f"工作空间目录已存在: {workspace_path}，尝试清理...")
        import shutil
        try:
            shutil.rmtree(workspace_path)
            print("清理成功")
        except Exception as e:
            print(f"清理失败: {e}")
            # 尝试直接打开
            # return de.open_workspace(workspace_path) # 如果支持的话
            # 暂时仍抛出异常或根据情况处理
            # raise RuntimeError(f"Workspace directory already exists: {workspace_path}")
            pass

    # 创建工作空间
    workspace = de.create_workspace(workspace_path)
    # 打开工作空间
    workspace.open()
    # 返回已打开的工作空间，完成后关闭
    return workspace
 
 
def create_a_library_and_add_it_to_the_workspace(workspace: de.Workspace):
    # assert workspace.path is not None
    # Libraries can only be added to an open workspace
    # 库只能添加到已打开的工作空间
    assert workspace.is_open
    # 在工作空间的目录中创建库
    library_path = workspace.path / library_name
    # 创建库
    de.create_new_library(library_name, library_path)
    # 将库添加到工作空间（更新lib.defs）
    workspace.add_library(library_name, library_path, de.LibraryMode.SHARED)
    lib = workspace.open_library(library_name, library_path, de.LibraryMode.SHARED)
    lib.setup_schematic_tech()
    return lib


ws = create_and_open_an_empty_workspace(workspace_path)
lib = create_a_library_and_add_it_to_the_workspace(ws)

# 使用工作空间的指针创建并添加库到空工作空间
def create_schematic(library: de.Library):
    design = db.create_schematic(f"{library_name}:{cell_name}:schematic")
 
    v = design.add_instance(
        ("ads_datacmps", "VAR", "symbol"), (3.5, -2.75), name="VAR1", angle=-90
    )
    assert v.is_var_instance
    for i in range(len(L)):
        ind = design.add_instance("ads_rflib:L:symbol", (i * 2, 0))
        ind.parameters["L"].value = f"L{i + 1} nH"
        ind.update_item_annotation()
        v.vars[f"L{i + 1}"] = f"{L[i]}"
        design.add_wire([(i * 2 + 1, 0), (i * 2 + 2, 0)])
 
    for i in range(len(C)):
        cap = design.add_instance("ads_rflib:C:symbol", (i * 2 + 1.5, -1), angle=-90)
        cap.parameters["C"].value = f"C{i + 1} pF"
        cap.update_item_annotation()
        v.vars[f"C{i + 1}"] = f"{C[i]}"
        design.add_wire([(i * 2 + 1.5, 0), (i * 2 + 1.5, -1)])
        design.add_instance("ads_rflib:GROUND:symbol", (i * 2 + 1.5, -2), angle=-90)
    del v.vars["X"]
 
    design.add_instance("ads_simulation:TermG:symbol", (-1, -1), angle=-90)
    design.add_wire([(-1, -1), (-1, 0), (0, 0)])
 
    design.add_instance("ads_simulation:TermG:symbol", (len(L) * 2 + 1, -1), angle=-90)
    design.add_wire([(len(L) * 2, 0), (len(L) * 2 + 1, 0.0)])
    design.add_wire([(len(L) * 2 + 1, 0), (len(L) * 2 + 1, -1.0)])
 
    sp = design.add_instance("ads_simulation:S_Param:symbol", (2, 2))
    sp.parameters["Start"].value = "0.01 GHz"
    sp.parameters["Stop"].value = f"{(fs * 2) / 1e9} GHz"
    sp.parameters["Step"].value = "0.01 GHz"
    sp.update_item_annotation()
    design.save_design()
    return design
 
 
# Create schematic with the lib object/pointer
design = create_schematic(lib)
 
netlist = design.generate_netlist()
simulator = ads.CircuitSimulator()
target_output_dir = os.path.join(workspace_path, "data")
simulator.run_netlist(netlist, output_dir=target_output_dir)
 
##### 数据处理和绘图 #####
 
ipython = getipython.get_ipython()
output_data = dataset.open(
    Path(os.path.join(target_output_dir, f"{cell_name}" + ".ds"))
)
 
# 将SP1.SP数据块转换为pandas数据帧
mydata = output_data["SP1.SP"].to_dataframe().reset_index()
 
# 提取数据并将S21和S11转换为dB
freq = mydata["freq"] / 1e6
S21 = 20 * np.log10(abs(mydata["S[2,1]"]))
S11 = 20 * np.log10(abs(mydata["S[1,1]"]))
 
# 使用matplotlib的内联绘图功能绘制结果
plt.xlabel("Frequency (MHz)")
plt.ylabel("S21 and S11 (dB)")
plt.grid(True)
plt.plot(freq, S21)
plt.plot(freq, S11)