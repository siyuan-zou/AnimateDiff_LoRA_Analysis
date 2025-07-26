import numpy as np
import pickle
import imageio
import os

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin

from cdfvd import fvd

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
prompt_list = [
    "masterpiece, best quality, highly detailed, ultra detailed, cherry blossom trees, petals falling, gentle breeze, spring season, tranquil park, golden sunlight, peaceful atmosphere, vibrant colors",
    "futuristic city, flying cars, neon glow, holograms, sci-fi, cyberpunk, high-tech buildings, rainy streets, reflections, ultra-detailed, cinematic lighting, masterpiece",
    "majestic snow-covered mountains, crystal-clear lake, morning mist, golden sunrise, serene landscape, pine trees, masterpiece, best quality, highly detailed, ultra realistic",
    "ancient temple ruins, overgrown vines, dense jungle, mystical atmosphere, glowing fireflies, moonlight filtering through trees, fantasy landscape, best quality, ultra detailed",
    "a cozy wooden cabin, deep in the snowy forest, warm light from windows, smoke rising from chimney, peaceful winter night, starry sky, soft snowflakes falling, high quality, ultra detailed",
    "deep underwater scene, glowing jellyfish, coral reef, vibrant marine life, sunlight filtering through the ocean, masterpiece, ultra realistic, high detail",
    "medieval castle, on top of a cliff, dramatic sky, storm approaching, lightning in the distance, cinematic atmosphere, ultra detailed, realistic, best quality",
    "a dragon soaring above a burning city, glowing red eyes, smoke and embers in the air, dramatic lighting, fantasy setting, ultra detailed, masterpiece",
    "a futuristic racing scene, high-speed hovercars, neon trails, cybernetic city backdrop, intense motion blur, ultra realistic, high detail, cinematic lighting",
    "a lone samurai standing under a cherry blossom tree, katana in hand, petals drifting in the wind, golden sunset, historical Japan, ultra detailed, masterpiece",
    "desert landscape, ancient ruins, hot sun, sand dunes, a lone traveler walking, long shadows, masterpiece, ultra detailed, cinematic composition",
    "a spaceship landing on an alien planet, strange glowing plants, massive moons in the sky, futuristic technology, sci-fi adventure, ultra realistic, high quality",
    "a giant whale swimming in the clouds, fantasy world, floating islands, warm golden sunlight, peaceful and dreamlike atmosphere, ultra detailed, masterpiece",
    "a Viking ship sailing through stormy seas, thunder and lightning, massive waves crashing, determined warriors on board, cinematic action, ultra realistic, high detail",
    "a neon-lit alleyway in a cyberpunk city, rain pouring down, reflections on the wet pavement, mysterious figure in a trench coat, atmospheric and moody, ultra detailed, cinematic",
    "a tranquil bamboo forest, soft morning light, mist rolling through, peaceful and zen-like atmosphere, ultra realistic, high detail, masterpiece",
    "an astronaut walking on Mars, vast red desert, distant mountains, dramatic lighting, cinematic science fiction, ultra realistic, masterpiece",
    "a mystical forest at twilight, bioluminescent plants, floating orbs of light, magical and otherworldly atmosphere, ultra detailed, best quality",
    "a luxury yacht sailing in a crystal blue ocean, clear sunny sky, gentle waves, people relaxing on deck, ultra realistic, high detail, cinematic",
    "a street festival at night, colorful lanterns, live performances, fireworks in the sky, people celebrating, ultra detailed, masterpiece, lively and vibrant atmosphere",
]


def lora_simple_block_combine(lora_list, block=0):
    combined_lora_list = []
    for lora_name_main in lora_list[:2]:
        dict1 = StableDiffusionLoraLoaderMixin.lora_state_dict(
            f"guoyww/animatediff-motion-lora-{lora_name_main}", cache_dir="./model/lora"
        )[0]
        for lora_name_sub in lora_list[2:]:
            dict2 = StableDiffusionLoraLoaderMixin.lora_state_dict(
                f"guoyww/animatediff-motion-lora-{lora_name_sub}", cache_dir="./model/lora"
            )[0]
            assert block >= 0 and block <= 4
            if block != 4:
                keys = (
                    list(dict1.keys())[block * 32 : (block + 1) * 32]
                    + list(dict1.keys())[-48 * (block + 1) : -48 * block]
                )
            else:
                keys = list(dict1.keys())[4 * 32 : 4 * 32 + 16]
            for key in keys:
                dict1[key] = dict2[key]
            new_name = lora_name_main + "-" + lora_name_sub
            os.makedirs(f"model/lora/block{block}", exist_ok=True)
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                f"model/lora/block{block}", dict1, weight_name=new_name + ".safetensors"
            )
            combined_lora_list.append(new_name)
    return combined_lora_list


def motion_baseline_generate(lora_list, prompt_list, samples_per_lora_per_prompt, device=torch.device("cuda")):
    output = {}
    for lora_name in lora_list:
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            cache_dir="./model/image_motion",
        ).to(device)
        output[lora_name] = {}
        pipe.load_lora_weights(
            "guoyww/animatediff-motion-lora-" + lora_name, adapter_name=lora_name, cache_dir="./model/lora"
        )
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        os.makedirs(f"data/generated/{lora_name}", exist_ok=True)
        for idx, prompt in enumerate(prompt_list):
            output[lora_name][idx] = []
            for i in range(samples_per_lora_per_prompt):
                output_unit = pipe(
                    height=256,
                    width=256,
                    prompt=prompt,
                    negative_prompt="bad quality, worse quality",
                    num_frames=16,
                    guidance_scale=7.5,
                    num_inference_steps=40,
                    output_type="pil",
                )
                output_content = output_unit.frames[0]
                export_to_video(output_content, f"data/generated/{lora_name}/prompt_{idx}_video_{i}.mp4")
                output[lora_name][idx].append(output_content)
    return output


def motion_combined_generate(
    combined_lora_list, prompt_list, block, samples_per_lora_per_prompt, device=torch.device("cuda")
):
    output = {}
    for lora_name in combined_lora_list:
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16, cache_dir="./model/domain_adapter"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            beta_schedule="linear",
            timestep_spacing="linspace",
            steps_offset=1,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            model_id,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            cache_dir="./model/image_motion",
        ).to(device)
        output[lora_name] = {}
        pipe.load_lora_weights(f"model/lora/block{block}/" + lora_name + ".safetensors", adapter_name=lora_name)
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        os.makedirs(f"data/generated/block{block}_{lora_name}", exist_ok=True)
        for idx, prompt in enumerate(prompt_list):
            output[lora_name][idx] = []
            for i in range(samples_per_lora_per_prompt):
                output_unit = pipe(
                    height=256,
                    width=256,
                    prompt=prompt,
                    negative_prompt="bad quality, worse quality",
                    num_frames=16,
                    guidance_scale=7.5,
                    num_inference_steps=40,
                    output_type="pil",
                )
                output_content = output_unit.frames[0]
                export_to_video(output_content, f"data/generated/block{block}_{lora_name}/prompt_{idx}_video_{i}.mp4")
                output[lora_name][idx].append(output_content)
    return output


def motion_combined_baseline_generate(
    lora_list, prompt_list, samples_per_lora_per_prompt, weights=[1.0, 1.0], device=torch.device("cuda")
):
    output = {}
    for lora_name_main in lora_list[:2]:
        for lora_name_sub in lora_list[2:]:
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16,
                cache_dir="./model/domain_adapter",
            ).to(device)
            scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                clip_sample=False,
                beta_schedule="linear",
                timestep_spacing="linspace",
                steps_offset=1,
            )
            pipe = AnimateDiffPipeline.from_pretrained(
                model_id,
                motion_adapter=adapter,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                cache_dir="./model/image_motion",
            ).to(device)
            pipe.load_lora_weights(
                "guoyww/animatediff-motion-lora-" + lora_name_main,
                adapter_name=lora_name_main,
                cache_dir="./model/lora",
            )
            pipe.load_lora_weights(
                "guoyww/animatediff-motion-lora-" + lora_name_sub, adapter_name=lora_name_sub, cache_dir="./model/lora"
            )
            pipe.set_adapters([lora_name_main, lora_name_sub], adapter_weights=weights)
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            lora_name = lora_name_main + "-" + lora_name_sub
            output[lora_name] = {}
            os.makedirs(f"data/generated/combine_baseline/{lora_name}", exist_ok=True)
            for idx, prompt in enumerate(prompt_list):
                output[lora_name][idx] = []
                for i in range(samples_per_lora_per_prompt):
                    output_unit = pipe(
                        height=256,
                        width=256,
                        prompt=prompt,
                        negative_prompt="bad quality, worse quality",
                        num_frames=16,
                        guidance_scale=7.5,
                        num_inference_steps=40,
                        output_type="pil",
                    )
                    output_content = output_unit.frames[0]
                    export_to_video(
                        output_content, f"data/generated/combine_baseline/{lora_name}/prompt_{idx}_video_{i}.mp4"
                    )
                    output[lora_name][idx].append(output_content)
    return output


if __name__ == "__main__":
    baseline = motion_baseline_generate(lora_list, prompt_list, 10)
    print("Baseline finish")
    combined_baseline = motion_combined_baseline_generate(
        ["pan-left", "pan-right", "tilt-up", "tilt-down"], prompt_list, 5
    )
    print("Combined baseline finish")
    for i in range(5):
        combined_lora_list = lora_simple_block_combine(["pan-left", "pan-right", "tilt-up", "tilt-down"], block=i)
        combined = motion_combined_generate(
            ["pan-left-tilt-down", "pan-left-tilt-up", "pan-right-tilt-up", "pan-right-tilt-down"], prompt_list, i, 5
        )
    print("All finish")
