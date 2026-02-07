from keysight.ads import de
from keysight.ads.de import db_uu as db
from keysight.edatoolbox import ads
import keysight.ads.dataset as dataset
import os
import matplotlib.pyplot as plt
from IPython.core import getipython
from pathlib import Path
import numpy as np
 
 
workspace_path = r"G:\wenlong\ADS\test_wrk"
cell_name = "test"
library_name = "test_lib"
 
 
def create_and_open_an_empty_workspace(workspace_path: str):
    # 确保没有已打开的工作空间
    if de.workspace_is_open():
        de.close_workspace()
 
    # 如果目录已存在，则无法创建工作空间
    if os.path.exists(workspace_path):
        raise RuntimeError(f"Workspace directory already exists: {workspace_path}")
 
    # 创建工作空间
    workspace = de.create_workspace(workspace_path)
    # 打开工作空间
    workspace.open()
    # 返回已打开的工作空间，完成后关闭
    return workspace
 

def create_a_library_and_add_it_to_the_workspace(workspace: de.Workspace):
    # assert workspace.path is not None
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
# 使用工作空间的指针创建并添加库到空工作空间
cap.parameters["C"].value = f"C{i + 1} pF"
cap.update_item_annotation()
design.add_wire([(i * 2 + 1.5, 0), (i * 2 + 1.5, -1)])
design.add_instance("ads_rflib:GROUND:symbol", (i * 2 + 1.5, -2), angle=-90)
 
design.add_instance("ads_simulation:TermG:symbol", (-1, -1), angle=-90)
design.add_instance("ads_simulation:TermG:symbol", (10, -1), angle=-90)
 
design.add_wire([(-1, -1), (-1, 0), (0, 0)])
design.add_wire([(10, -1), (10, 0)])
 
sp = design.add_instance("ads_simulation:S_Param:symbol", (0, 2))
sp.parameters["Start"].value = "0.01 GHz"
sp.parameters["Stop"].value = "0.5 GHz"
sp.parameters["Step"].value = "0.001 GHz"
sp.update_item_annotation()
design.save_design()

 
 
# 使用lib对象/指针创建原理图
design = create_schematic(lib)
 
##### VAR定义 #####
 
L_values = ["100", "40", "100", "40", "100"]
C_values = ["30", "10", "30", "10"]
 
var_inst = design.add_instance(
    ("ads_datacmps", "VAR", "symbol"), (3.5, 1.875), name="VAR1", angle=90
)
for val in range(len(L_values)):
    var_inst.vars[f"L{val + 1}"] = L_values[val]
del var_inst.vars["X"]
 
var_inst = design.add_instance(
    ("ads_datacmps", "VAR", "symbol"), (5, 1.875), name="VAR2", angle=90
)
for val in range(len(C_values)):
    var_inst.vars[f"C{val + 1}"] = C_values[val]
del var_inst.vars["X"]
 
##### 测量方程块 #####
eq_list = [
    "groupdelay=(-1/360)*diff(unwrap(phase(S(2,1))))/diff(freq)",
    "s21mag=mag(S(2,1))",
    "s21phase=phase(S(2,1))",
]
 
 
def add_measeqn(design, eq_name, eq_list):
    # add MeasEqn to the schmatic
    measeqn = design.add_instance(
        ("ads_simulation", "MeasEqn", "symbol"), (6.5, 1.875), name=eq_name, angle=-90
    )
    # change first existing equation
    measeqn.parameters["Meas"].value = [eq_list[0]]
    # add new equations with rest of equation list
    for i in range(len(eq_list) - 1):
        measeqn.parameters["Meas"].repeats.append(
            db.ParamItemString("Meas", "SingleTextLine", eq_list[i + 1])
        )
    measeqn.update_item_annotation()
 
 
add_measeqn(design, "Meas1", eq_list)
 
### 网表创建和仿真 ###
 
netlist = design.generate_netlist()
simulator = ads.CircuitSimulator()
target_output_dir = os.path.join(workspace_path, "data")
simulator.run_netlist(netlist, output_dir=target_output_dir)
 
##### 数据处理和绘图 #####
 
output_data = dataset.open(
    Path(os.path.join(target_output_dir, f"{cell_name}" + ".ds"))
)
 
# 检查数据集中可用的数据块
print("Available Data Blocks: ", output_data.varblock_names)
 
# 查找包含我们结果的相关数据块
for datablock in output_data.find_varblocks_with_var_name("groupdelay"):
    print("Group Delay expression is found in:", datablock.name)
    gd = datablock.name
 
for datablock in output_data.find_varblocks_with_var_name("S[2,1]"):
    print("S21 measurement is found in:", datablock.name)
    sp = datablock.name
 
# 将SP1.SP数据块转换为pandas数据帧
mydata = output_data[sp].to_dataframe().reset_index()
# 将群延迟数据块转换为pandas数据帧
mygd = output_data[gd].to_dataframe().reset_index()
 
# 提取数据并将S21和S11转换为dB
freq = mydata["freq"] / 1e6
S21 = 20 * np.log10(abs(mydata["S[2,1]"]))
S11 = 20 * np.log10(abs(mydata["S[1,1]"]))
 
# 使用matplotlib的内联绘图功能绘制结果
ipython = getipython.get_ipython()
ipython.run_line_magic("matplotlib", "inline")
_, ax = plt.subplots()
ax.set_title("Python Filter Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("S21 and S11 (dB)")
plt.grid(True)
plt.plot(freq, S21)
plt.plot(freq, S11)
 
# 使用matplotlib的内联绘图功能绘制群延迟结果
freq = mygd["freq"] / 1e6
groupdelay = mygd["groupdelay"] / 1e-9
 
ipython = getipython.get_ipython()
ipython.run_line_magic("matplotlib", "inline")
_, ax = plt.subplots()
ax.set_title("Filter Group Delay Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Group Delay (nsec)")
plt.grid(True)
plt.plot(freq, groupdelay)