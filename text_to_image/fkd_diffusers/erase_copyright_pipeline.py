import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
# Added for FK Steering
from fkd_class import FKD
from rewards import get_reward_function
from w2s_pipeline_sd import esd_w2s_StableDiffusion
import torch
import torch.nn.functional as F
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl import (
    StableDiffusionXLPipelineOutput,
    StableDiffusionXLPipeline,
)

import numpy as np
from fkd_pipeline_sdxl_inverse import FKDStableDiffusionXL
from w2s_pipeline_sd import esd_w2s_StableDiffusion
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import argparse
from diffusers import DDIMScheduler,DDIMInverseScheduler

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    strong_pipeline = FKDStableDiffusionXL.from_pretrained(args.strong_model_name,torch_dtype=torch.float32)
    strong_pipeline.scheduler = DDIMScheduler.from_config(strong_pipeline.scheduler.config)
    weak_pipeline = esd_w2s_StableDiffusion.from_pretrained(
        args.weak_model_name,
        unet_safe_path = args.unet_safe_path,
        torch_dtype=torch.float16,
    )
    weak_pipeline.scheduler = DDIMScheduler.from_config(weak_pipeline.scheduler.config)
    weak_pipeline.vae.to(torch.float32)

    weak_pipeline.to(device)
    strong_pipeline.to(device)

    stamps_a , stamps_b, stamps_c, num_inference_step = args.time_stamps[0], args.time_stamps[1], args.time_stamps[2], args.time_stamps[3]
    print(f"stamps_a: {stamps_a}, stamps_b: {stamps_b}, stamps_c: {stamps_c}, num_inference_step: {num_inference_step}")

    # stage 1 build strong model image condition
    output = strong_pipeline.partial_denoise(
        prompt=args.prompt,
        num_inference_steps=num_inference_step,
        eta=1.0,
        seed=args.seed,
        start_timesteps=stamps_a,
        end_timesteps=stamps_b,
    )
    output.images[0].save(f"/home/ubuntu/tiansheng/Fk-Diffusion-Steering/final_result/w2s/strong_output_{stamps_b}.png")
    print('stage 1 done, generated strong model image condition')

    # stage 2 build weak model image condition
    input_img = weak_pipeline.image_processor.preprocess(output.images[0],height = 768,width =768).to('cuda')
    posterior = weak_pipeline.vae.encode(input_img).latent_dist
    latents = posterior.mean * weak_pipeline.vae.config.scaling_factor
    print('Latents shape:', latents.shape)
    output = weak_pipeline.partial_denoise(
            args.prompt,
            num_inference_steps=num_inference_step,
            eta=1.0,
            seed=args.seed,
            start_timesteps=stamps_b,
            end_timesteps=stamps_c,
            latents=latents.to(torch.float16),
            use_safe = args.use_safe,
        )
    output.images[0].save(f"/home/ubuntu/tiansheng/Fk-Diffusion-Steering/final_result/w2s/weak_output_{stamps_c}.png")
    print('stage 2 done, generated weak model image condition')

    input_img = weak_pipeline.image_processor.preprocess(output.images[0],height = 1024,width =1024).to('cuda')
    posterior = strong_pipeline.vae.encode(input_img).latent_dist
    latents = posterior.mean * weak_pipeline.vae.config.scaling_factor
    output = strong_pipeline.partial_denoise(
        args.prompt,
        num_inference_steps= num_inference_step,
        eta=1.0,
        start_timesteps=stamps_c,
        end_timesteps=num_inference_step,
        latents=latents,
    )
    output.images[0].save(f"/home/ubuntu/tiansheng/Fk-Diffusion-Steering/final_result/w2s/final_output_{num_inference_step}.png")
    print('Successfully generated image')
    print('hello')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strong_model_name",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0'
    )
    parser.add_argument(
        "--unet_safe_path",
        type=str,
        default='/home/ubuntu/tiansheng/Fk-Diffusion-Steering/text_to_image/alignment_methods/esd-ironman_from_ironman-xattn_1-epochs_200.pt',
        help="safe unet path",
    )
    parser.add_argument(
        "--weak_model_name",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        help="model_name",
    )
    parser.add_argument(
        "--use_safe",
        type=bool,
        default=True,
        help="whether to use safe unet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=912,
        help="random seed",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of iron man eatting a burger",
        help="prompt",
    )
    parser.add_argument(
        "--time_stamps",
        type=list,
        default=[0,10,20,100],
        help="time_stamps",
    )
    args = parser.parse_args()
    main(args)