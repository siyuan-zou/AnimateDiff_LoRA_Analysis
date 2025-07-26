import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_video
import os
import time
# import h5py
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 256] into [-1, 1]
            T.Resize(size=(256, 256)),
        ]
    )
    batch = transforms(batch)
    return batch

def generate_DLOW(video_path):
    print(f"正在处理视频 {video_path} 的光流数据...")
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    frames, _, _ = read_video(str(video_path))
    frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    print(f"视频 {video_path} 的帧数：{frames.shape}")

    img1_batch = torch.stack([frames[i] for i in range(15)])
    img2_batch = torch.stack([frames[i] for i in range(1,16)])

    img1_batch = preprocess(img1_batch).to(device)
    img2_batch = preprocess(img2_batch).to(device)

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = list_of_flows[-1] # dtype = torch.float32, shape = torch.Size([15, 2, 256, 256]) = (N, 2, H, W)

    return predicted_flows


def save_flows_to_npy(video_flows, output_path):
    """
    保存光流数据为 .npy 格式
    """
    # 保存所有光流数据到一个 npy 文件
    np.save(output_path, video_flows)
    print(f"光流数据已保存到 {output_path}")

# def save_flows_to_hdf5(video_flows, output_path):
#     """
#     保存光流数据为 .h5 格式
#     """
#     with h5py.File(output_path, 'w') as f:
#         f.create_dataset('flows', data=video_flows)
#     print(f"光流数据已保存到 {output_path}")

def main(videos_path: str, output_dir: str):
    all_flows = []  # 用于存储所有视频的光流数据

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        
        # 获取每个视频的光流数据
        predicted_flows = generate_DLOW(video_path)
        
        # 将每个视频的光流数据存储为 NumPy 数组（形状: [15, 2, 256, 256]）
        video_flows = predicted_flows.detach().cpu().numpy()  # 转为 NumPy 数组
        all_flows.append(video_flows)
        
        # 保存每个视频的光流数据
        video_name = os.path.splitext(video)[0]
        print(f"正在处理 {video_name} 的光流数据...")
        # output_path = os.path.join(output_dir, f"{video_name}_flows.npy")
        # save_flows_to_npy(video_flows, output_path)

    # 如果你希望将所有视频的光流数据存储到一个文件中
    all_flows = np.array(all_flows)  # 转为 NumPy 数组，形状：[num_videos, 15, 2, 256, 256]
    all_flows_output_path = os.path.join(output_dir, 'all_videos_flows.npy')
    save_flows_to_npy(all_flows, all_flows_output_path)

if __name__ == '__main__':
    # # 调用 main 函数处理视频并保存数据
    # block = 4

    # for lora in combined_lora_list:
    #     videos_path = f'./data/generated/block{block}_{lora}'
    #     output_dir = f'./DLOW/generated/block{block}_{lora}'  # 设置你想存储光流数据的输出目录
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     start = time.time()
    #     main(videos_path, output_dir)
    #     print(f"处理 {lora} 的光流数据用时：{time.time() - start:.2f} 秒")

    for lora in lora_list[2:]:
        videos_path = f'./data/generated/{lora}'
        output_dir = f'./DLOW/generated/{lora}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        start = time.time()
        main(videos_path, output_dir)
        print(f"处理 {lora} 的光流数据用时：{time.time() - start:.2f} 秒")


    

