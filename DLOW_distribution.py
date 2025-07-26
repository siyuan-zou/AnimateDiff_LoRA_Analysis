import torch
from sklearn.decomposition import PCA
import numpy as np
import os
import time

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

def generate_distribution(dlows_path: str, n_components: int = 2):
    """
    生成所有视频的低维表示
    """
    all_video_dlows = np.load(dlows_path)
    print("all_dlows shape: ", all_video_dlows.shape) # (num_videos, 15, 2, 256, 256)

    all_video_dlow = []  # 用于存储所有视频的低维表示

    for flows in all_video_dlows:

        avg_flow = np.mean(flows, axis=0)  # 计算每个视频的平均光流，得到 (2, 256, 256)
        
        # 将光流展平为一维向量（2 * 256 * 256维）
        avg_flow_flattened = avg_flow.reshape(-1)
        
        # 保存这个视频的低维表示
        all_video_dlow.append(avg_flow_flattened)

    # 假设 video_flows 是 (100, 2, 256, 256) 的光流数据
    print("all_video_dlow shape: ", np.array(all_video_dlow).shape)  # (100, 2 * 256 * 256)

    # **Step 1: 训练 PCA (Fit)**
    pca = PCA(n_components=n_components)
    pca.fit(all_video_dlow)  # 只需要在所有数据上拟合一次

    # **Step 2: 用训练好的 PCA 转换所有数据 (Transform)**
    reduced_flows = pca.transform(all_video_dlow)  # (100, 2)
    print("reduced_flows shape: ", reduced_flows.shape)  # (100, 2)

    return reduced_flows

def save_distribution_to_npy(reduced_flows, output_path):
    """
    保存低维表示为 .npy 格式
    """
    # 保存所有低维表示到一个 npy 文件
    np.save(output_path, reduced_flows)
    print(f"低维表示已保存到 {output_path}")

def main(dlows_path: str, output_dir: str, n_components: int = 2):
    
    for lora in lora_list[2:]: # block4
        path = lora

        start = time.time()
        print(f"正在处理 {path} 的低维表示...")
        lora_dlows_path = os.path.join(dlows_path, f"{path}/all_videos_flows.npy")
        
        # 获取每个视频的光流数据
        reduced_flows = generate_distribution(lora_dlows_path, n_components=n_components)
        
        # 保存每个视频的低维表示
        output_path = os.path.join(dlows_path, f"{path}/reduced_dlow_{n_components}.npy")
        save_distribution_to_npy(reduced_flows, output_path)

        print(f"处理 {path} 的低维表示用时：{time.time() - start:.2f} 秒")

if __name__ == '__main__':
    main('./DLOW/generated', './DLOW/generated', 10)
