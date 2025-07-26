import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def save_distribution_to_npy(reduced_flows, output_path):
    """
    保存低维表示为 .npy 格式
    """
    # 保存所有低维表示到一个 npy 文件
    np.save(output_path, reduced_flows)
    print(f"低维表示已保存到 {output_path}")

def generate_camera_trajectory(flow: np.ndarray) -> np.ndarray:
    """
    生成相机轨迹
    :param flow: 光流，形状 (15, 2, 256, 256)
    :return: 相机轨迹，形状 (15, 9)
    """

    camera_trajectory = []

    for i in range(15):  # 16 帧，有 15 组光流
        flow_uv = flow[i]  # 形状 (2, 256, 256)
        h, w = flow_uv.shape[1:]

        # 生成网格点
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        pts1 = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
        pts2 = pts1 + flow_uv.reshape(2, -1).T  # 应用光流偏移

        # 计算单应性矩阵
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        if H is not None:
            camera_trajectory.append(H.flatten())

    camera_trajectory = np.array(camera_trajectory)  # (15, 9)
    return camera_trajectory

def main(dlows_path: str):
    
    for path in os.listdir(dlows_path):

        start = time.time()
        print(f"正在处理 {path} 的Homography表示...")
        lora_dlows_path = os.path.join(dlows_path, f"{path}/all_videos_flows.npy")
        
        # 获取每个视频的光流数据
        reduced_camera_trajectory = []
        for flow in np.load(lora_dlows_path):
            camera_trajectory = generate_camera_trajectory(flow)
            camera_trajectory = camera_trajectory.mean(axis=0) # 计算每个视频的平均相机轨迹，得到 (9,)
            # print(camera_trajectory.shape)
            reduced_camera_trajectory.append(camera_trajectory)
        
        reduced_camera_trajectory = np.array(reduced_camera_trajectory)  # (num_videos, 9)
        # 保存每个视频的低维表示
        output_path = os.path.join(dlows_path, f"{path}/reduced_camera_trajectory.npy")
        save_distribution_to_npy(reduced_camera_trajectory, output_path)

        print(f"处理 {path} 的低维表示用时：{time.time() - start:.2f} 秒")

if __name__ == '__main__':
    dlows_path = "DLOW/generated"

    main(dlows_path)