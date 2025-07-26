import numpy as np
import pickle
import imageio

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin

import time
import os


def lora_simple_block_combine(lora_list, block = 0):
    combined_lora_list = []
    for lora_name_main in lora_list[:2]:
        dict1 = StableDiffusionLoraLoaderMixin.lora_state_dict(f"guoyww/animatediff-motion-lora-{lora_name_main}", cache_dir="./model/lora")[0]
        for lora_name_sub in lora_list[2:]:
            dict2 = StableDiffusionLoraLoaderMixin.lora_state_dict(f"guoyww/animatediff-motion-lora-{lora_name_sub}", cache_dir="./model/lora")[0]
            assert block >= 0 and block <=4
            if block != 4:
                keys = list(dict1.keys())[block*32:(block+1)*32] + list(dict1.keys())[-48*(block+1):-48*block]
            else:
                keys = list(dict1.keys())[4*32:4*32+16]
            for key in keys:
                dict1[key] = dict2[key]
            new_name = lora_name_main + "-" + lora_name_sub
            StableDiffusionLoraLoaderMixin.save_lora_weights(f"model/lora/block{block}", dict1, weight_name = new_name + ".safetensors")
            combined_lora_list.append(new_name)
    return combined_lora_list


def motion_simple(lora_name : str, prompt : str, sample_num :int = 1, device = torch.device("cuda")):
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter").to(device)
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16, cache_dir="./model/image_motion").to(device)
    pipe.load_lora_weights("guoyww/animatediff-motion-lora-" + lora_name, adapter_name=lora_name, cache_dir="./model/lora") 
    scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,)
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    output = []
    for num in range(sample_num):
        output_unit = pipe(
            height = 256,
            width = 256,
            prompt = prompt,
            negative_prompt="bad quality, worse quality",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=30,
            output_type = "pil",)
        #'np', 'pt', 'pil'
        output.append(output_unit.frames[0])
    # for i in range(sample_num):
    #     np.save(f"data/generated/{lora_name}/prompt_1_array_{i}.npy", output[i])
    for i in range(sample_num):
        export_to_video(output[i], f"data/generated/mp4/{lora_name}/prompt_1_video_{i}.mp4")
    return output_unit

def motion_baseline_generate(lora_list, prompt_list, samples_per_lora_per_prompt, device = torch.device("cuda")):
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter").to(device)
    scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,)
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, scheduler = scheduler, torch_dtype=torch.float16, cache_dir="./model/image_motion").to(device)
    output = {}
    for lora_name in lora_list:
        output[lora_name] = {}
        pipe.load_lora_weights("guoyww/animatediff-motion-lora-" + lora_name, adapter_name=lora_name, cache_dir="./model/lora")
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        for idx,prompt in enumerate(prompt_list):
            output[lora_name][idx] = []
            for i in range(samples_per_lora_per_prompt):
                output_unit = pipe(
                    height = 256,
                    width = 256,
                    prompt = prompt,
                    negative_prompt="bad quality, worse quality",
                    num_frames=16,
                    guidance_scale=7.5,
                    num_inference_steps=40,
                    output_type = "pil",)
                output_content = output_unit.frames[0]

                os.makedirs(f"data/generated/{lora_name}", exist_ok=True)  # 如果目录不存在，则创建
                export_to_video(output_content, f"data/generated/{lora_name}/prompt_{idx}_video_{i}.mp4")
                output[lora_name][idx].append(output_content)
    return output
                
def motion_combined_generate(combined_lora_list, prompt_list, block, samples_per_lora_per_prompt, device = torch.device("cuda")):
    '''
    generate videos with block-layerly combined motion loras
    '''
    

    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter").to(device)
    scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,)
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, scheduler = scheduler, torch_dtype=torch.float16, cache_dir="./model/image_motion").to(device)
    output = {}
    for lora_name in combined_lora_list:
        output[lora_name] = {}
        pipe.load_lora_weights(f"model/lora/block{block}/" + lora_name + ".safetensors", adapter_name=lora_name)
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        for idx,prompt in enumerate(prompt_list):
            output[lora_name][idx] = []
            for i in range(samples_per_lora_per_prompt):
                # start_time = time.time()
                output_unit = pipe(
                    height = 256,
                    width = 256,
                    prompt = prompt,
                    negative_prompt="bad quality, worse quality",
                    num_frames=16,
                    guidance_scale=7.5,
                    num_inference_steps=40,
                    output_type = "pil",)
                output_content = output_unit.frames[0]

                output_dir = f"data/generated/block{block}_{lora_name}"
                os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，则创建

                export_to_video(output_content, f"{output_dir}/prompt_{idx}_video_{i}.mp4")
                # export_to_video(output_content, f"data/generated/block{block}_{lora_name}/prompt_{idx}_video_{i}.mp4")
                output[lora_name][idx].append(output_content)
                # print(f"A video takes time: {time.time()-start_time}")
    return output

def motion_combined_baseline_generate(lora_list, prompt_list, samples_per_lora_per_prompt, weights = [1.0,1.0], device = torch.device("cuda")):
    '''
    generate videos with officially combined motion loras(scalar weight sum)
    '''
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter").to(device)
    scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,)
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, scheduler = scheduler, torch_dtype=torch.float16, cache_dir="./model/image_motion").to(device)
    output = {}
    for lora_name_main in lora_list[:2]:
        pipe.load_lora_weights("guoyww/animatediff-motion-lora-" + lora_name_main, adapter_name=lora_name_main, cache_dir="./model/lora")
        for lora_name_sub in lora_list[2:]:
            pipe.load_lora_weights("guoyww/animatediff-motion-lora-" + lora_name_sub, adapter_name=lora_name_sub, cache_dir="./model/lora")
            pipe.set_adapters([lora_name_main, lora_name_sub], adapter_weights=weights)
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            lora_name = lora_name_main + "-" + lora_name_sub
            output[lora_name] = {}
            for idx,prompt in enumerate(prompt_list):
                output[lora_name][idx] = []
                for i in range(samples_per_lora_per_prompt):
                    output_unit = pipe(
                        height = 256,
                        width = 256,
                        prompt = prompt,
                        negative_prompt="bad quality, worse quality",
                        num_frames=16,
                        guidance_scale=7.5,
                        num_inference_steps=40,
                        output_type = "pil",)
                    output_content = output_unit.frames[0]

                    output_dir = f"data/generated/combine_baseline/{lora_name}"
                    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，则创建

                    export_to_video(output_content, f"{output_dir}/prompt_{idx}_video_{i}.mp4")
                    # export_to_video(output_content, f"data/generated/combine_baseline/{lora_name}/prompt_{idx}_video_{i}.mp4")
                    output[lora_name][idx].append(output_content)
    return output
            