# AnimateDiff LoRA Combination Sensitivity and Generation Quality Analysis

With the rise of Text-to-Image (T2I) models like Stable Diffusion, personalized visual content generation has advanced significantly, particularly with the introduction of techniques such as LoRA. However, most T2I models are limited to static images. AnimateDiff addresses this gap by enabling time-consistent video generation based on T2I models. In this project, we explored AnimateDiff's MotionLoRA mechanism, focusing on how motion effects can be combined through two methods: scalar-weighted LoRA summation and symmetrical single-layer replacement. Using eight official MotionLoRA weights, we conducted large-scale video generation and evaluated the sensitivity of different UNet layers to motion control. Our results provide insights into motion effect composition and offer guidance on optimizing LoRA-based video generation.

- Files named by Camera_..., ..._analysis, motion estimation and DLOW... are used for sensitivity part, with other reducing methods and utils.

- Quality_evaluate.py is for generic video generation quality measurement, with preliminary_test as basic functions test and some trials.

- Sensitivity results in "output_pics", quality results in "quality results".

Contributer:
Yingwei T.
