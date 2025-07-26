import numpy as np
import matplotlib.pyplot as plt

lora_list = ["zoom-in", "zoom-out", "pan-left", "pan-right", "tilt-up", "tilt-down", "rolling-clockwise", "rolling-anticlockwise"]

combined_lora_list = ['zoom-in-pan-left',
 'zoom-in-pan-right',
 'zoom-in-tilt-up',
 'zoom-in-tilt-down',
 'zoom-in-rolling-clockwise',
 'zoom-in-rolling-anticlockwise',
 'zoom-out-pan-left',
 'zoom-out-pan-right',
 'zoom-out-tilt-up',
 'zoom-out-tilt-down',
 'zoom-out-rolling-clockwise',
 'zoom-out-rolling-anticlockwise']

 def load_dlow(block: int, main_lora: str, sub_lora: str):
    """
    加载指定 block 和 lora 的 DLOW 数据
    """
    # 读取 DLOW 数据
    dlow_path = f"../DLOW/generated/{block}_{main_lora}-{sub_lora}/reduced_dlow.npy"
    dlow = np.load(dlow_path)
    return dlow

def plot_dlow(dlow):
    """
    绘制 DLOW 数据的散点图
    """
    plt.scatter(dlow[:, 0], dlow[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('DLOW Distribution')
    plt.show()

def main():
    


