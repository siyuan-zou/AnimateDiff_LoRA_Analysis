import pandas as pd
from cdfvd import fvd
import os

from torch.utils.data import ConcatDataset, DataLoader

lora_list = [
    "zoom-in",
    "zoom-out",
    "pan-left",
    "pan-right",
    "tilt-up",
    "tilt-down",
    "rolling-clockwise",
    "rolling-anticlockwise",
]
combined_lora_list = [
    "zoom-in-pan-left",
    "zoom-in-pan-right",
    "zoom-in-tilt-up",
    "zoom-in-tilt-down",
    "zoom-in-rolling-clockwise",
    "zoom-in-rolling-anticlockwise",
    "zoom-out-pan-left",
    "zoom-out-pan-right",
    "zoom-out-tilt-up",
    "zoom-out-tilt-down",
    "zoom-out-rolling-clockwise",
    "zoom-out-rolling-anticlockwise",
    "pan-left-tilt-up",
    "pan-left-tilt-down",
    "pan-right-tilt-up",
    "pan-right-tilt-down",
]


def individual_evaluate(lora_name_list, combined_baseline_lora_list=None, block_list=None):
    score_dict = {}
    os.makedirs("data/result/individual", exist_ok=True)
    if not block_list:
        assert combined_baseline_lora_list
        evaluator = fvd.cdfvd("videomae", ckpt_path="model/cdfvd_evaluator/videomae.pth", n_fake="full")
        evaluator.load_videos("sky", data_type="stats_pkl", resolution=256, sequence_length=16)
        for lora_name in lora_name_list:
            evaluator.compute_fake_stats(
                evaluator.load_videos(video_info=f"data/generated/{lora_name}", data_type="video_folder")
            )
            score = evaluator.compute_fvd_from_stats()
            score_dict[lora_name] = [score]
        for combined_lora_name in combined_baseline_lora_list:
            evaluator.compute_fake_stats(
                evaluator.load_videos(
                    video_info=f"data/generated/combine_baseline/{combined_lora_name}", data_type="video_folder"
                )
            )
            score = evaluator.compute_fvd_from_stats()
            score_dict[combined_lora_name] = [score]
        df = pd.DataFrame(score_dict)
        df.to_csv("data/result/individual/baseline.csv", index=False)
    else:
        assert not combined_baseline_lora_list
        evaluator = fvd.cdfvd("videomae", ckpt_path="model/cdfvd_evaluator/videomae.pth", n_fake="full")
        evaluator.load_videos("sky", data_type="stats_pkl", resolution=256, sequence_length=16)
        for block in block_list:
            score_dict[block] = {}
            for lora_name in lora_name_list:
                evaluator.compute_fake_stats(
                    evaluator.load_videos(
                        video_info=f"data/generated/block{block}_{lora_name}", data_type="video_folder"
                    )
                )
                score = evaluator.compute_fvd_from_stats()
                score_dict[block][lora_name] = [score]
        df = pd.DataFrame(score_dict)
        df.to_csv("data/result/individual/block_combined.csv", index=False)
    return df


def sum_loaders(loaders, batch_size=16, shuffle=True, num_workers=4):
    datasets = [loader.dataset for loader in loaders]
    sum_dataset = ConcatDataset(datasets)
    final_loader = DataLoader(sum_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return final_loader


def general_evaluate(lora_name_list, combined_baseline_lora_list=None, block_list=None):
    score_dict = {}
    os.makedirs("data/result/general", exist_ok=True)
    if not block_list:
        assert combined_baseline_lora_list
        evaluator = fvd.cdfvd("videomae", ckpt_path="model/cdfvd_evaluator/videomae.pth", n_fake="full")
        evaluator.load_videos("sky", data_type="stats_pkl", resolution=256, sequence_length=16)

        loaders_baseline = [
            evaluator.load_videos(video_info=f"data/generated/{lora_name}", data_type="video_folder")
            for lora_name in lora_name_list
        ]
        # datasets_baseline = [loader.dataset for loader in loaders_baseline]
        # dataset_baseline = ConcatDataset(datasets_baseline)
        # final_loader_baseline = DataLoader(dataset_baseline, batch_size=16, shuffle=True, num_workers=4)
        final_loader_baseline = sum_loaders(loaders_baseline)

        loaders_combined_baseline = [
            evaluator.load_videos(
                video_info=f"data/generated/combine_baseline/{combined_lora_name}", data_type="video_folder"
            )
            for combined_lora_name in combined_baseline_lora_list
        ]
        # datasets_combined_baseline = [loader.dataset for loader in loaders_combined_baseline]
        # dataset_combined_baseline = ConcatDataset(datasets_combined_baseline)
        # final_loader_combined_baseline = DataLoader(dataset_combined_baseline, batch_size=16, shuffle=True, num_workers=4)
        final_loader_combined_baseline = sum_loaders(loaders_combined_baseline)

        evaluator.compute_fake_stats(final_loader_baseline)
        score = evaluator.compute_fvd_from_stats()
        score_dict["Baseline"] = [score]

        evaluator.compute_fake_stats(final_loader_combined_baseline)
        score = evaluator.compute_fvd_from_stats()
        score_dict["Combined_baseline"] = [score]

        df = pd.DataFrame(score_dict)
        df.to_csv("data/result/general/all_baseline.csv", index=False)

    else:
        assert not combined_baseline_lora_list
        evaluator = fvd.cdfvd("videomae", ckpt_path="model/cdfvd_evaluator/videomae.pth", n_fake="full")
        evaluator.load_videos("sky", data_type="stats_pkl", resolution=256, sequence_length=16)
        for block in block_list:
            loaders = [
                evaluator.load_videos(video_info=f"data/generated/block{block}_{lora_name}", data_type="video_folder")
                for lora_name in lora_name_list
            ]
            final_loader = sum_loaders(loaders)
            evaluator.compute_fake_stats(final_loader)
            score = evaluator.compute_fvd_from_stats()
            score_dict[block] = [score]
        df = pd.DataFrame(score_dict)
        df.to_csv("data/result/general/all_block_combined.csv")

    return df


if __name__ == "__main__":
    individual_scores_1 = individual_evaluate(lora_list, combined_lora_list)
    individual_scores_2 = individual_evaluate(combined_lora_list, block_list=list(range(5)))
    general_scores_1 = general_evaluate(lora_list, combined_lora_list)
    general_scores_2 = general_evaluate(combined_lora_list, block_list=list(range(5)))
