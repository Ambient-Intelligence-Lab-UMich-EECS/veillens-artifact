"""
Stage 3 (editing), generation half.

`ControlNetInpainter` wraps SDXL inpainting with a depth ControlNet. `inpaint_batch` runs a
whole batch of crops through a single forward pass.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
)
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: SDXL wants dimensions that are multiples of this.
_ROUND_TO = 8


class ControlNetInpainter:
    """Handles SDXL inpainting with a depth ControlNet."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['detection']['device']

        print("\n" + "=" * 70)
        print("Loading Diffusion Models (this may take a while on first run)")
        print("=" * 70)

        print("\n[1/2] Loading ControlNet Depth (~1.5GB on first download)...")
        depth_start = time.time()
        controlnet = ControlNetModel.from_pretrained(
            config['models']['controlnet_depth'],
            torch_dtype=torch.float16 if config['performance']['use_fp16'] else torch.float32
        )
        print(f"      ✓ ControlNet Depth loaded in {time.time() - depth_start:.1f}s")

        # Create the SDXL pipeline with the depth ControlNet
        print("\n[2/2] Loading SDXL ControlNet Inpainting Pipeline (~6.9GB on first download)...")
        print(f"      Model: {config['models']['sdxl_base_model']}")
        sdxl_start = time.time()
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            config['models']['sdxl_base_model'],
            controlnet=controlnet,
            torch_dtype=torch.float16 if config['performance']['use_fp16'] else torch.float32,
            variant="fp16" if config['performance']['use_fp16'] else None,
        )
        print(f"      ✓ SDXL Inpainting loaded in {time.time() - sdxl_start:.1f}s")

        print(f"\n      Moving models to {self.device}...")
        device_start = time.time()
        self.pipe.to(self.device)
        print(f"      ✓ Models ready on device in {time.time() - device_start:.1f}s")

        # Enable memory optimizations
        print("\n      Applying memory optimizations...")

        if config['performance']['enable_attention_slicing']:
            self.pipe.enable_attention_slicing()
            print("      ✓ Attention slicing enabled")

        if config['performance']['enable_vae_tiling']:
            self.pipe.enable_vae_tiling()
            print("      ✓ VAE tiling enabled (saves ~40% memory for large images)")

        print("      ✓ All optimizations applied")

    def _to_pil_rgb(self, img):
        """Convert numpy array or PIL to RGB PIL Image."""
        if isinstance(img, np.ndarray):
            return Image.fromarray(img).convert('RGB')
        return img.convert('RGB') if img.mode != 'RGB' else img

    def _to_pil_l(self, img):
        """Convert numpy array or PIL to grayscale PIL Image."""
        if isinstance(img, np.ndarray):
            return Image.fromarray(img).convert('L')
        return img.convert('L') if img.mode != 'L' else img

    def inpaint_batch(
        self,
        images: list,
        masks: list,
        depth_maps: list,
        prompts: list,
        negative_prompts: list
    ) -> list:
        """
        Perform batch inpainting in a single forward pass.

        Args:
            images: List of numpy arrays or PIL Images (cropped images) - MUST be same size
            masks: List of numpy arrays or PIL Images (cropped masks) - MUST be same size
            depth_maps: List of numpy arrays or PIL Images (cropped depth maps) - MUST be same size
            prompts: List of positive prompts (one per image)
            negative_prompts: List of negative prompts (one per image)

        Returns:
            List of inpainted images as numpy arrays (RGB uint8)
        """
        gen_config = self.config['generation']
        batch_size = len(images)

        print(f"[BATCH] Processing {batch_size} clues in parallel (true batching)")

        # 1. Convert all inputs to PIL format
        images_pil = [self._to_pil_rgb(img) for img in images]
        masks_pil = [self._to_pil_l(mask) for mask in masks]

        # 2. Prepare control images
        control_images = [self._to_pil_rgb(depth) for depth in depth_maps]

        # 3. Apply the style prefix that every generation shares
        positive_prefix = gen_config.get('positive_prompt_prefix', '')
        full_prompts = [f"{positive_prefix}, {p}" if positive_prefix else p for p in prompts]
        full_negative_prompts = list(negative_prompts)

        # 4. Get and validate dimensions
        orig_width, orig_height = images_pil[0].size
        for i, img in enumerate(images_pil[1:], 1):
            if img.size != (orig_width, orig_height):
                raise ValueError(
                    f"Batch dimension mismatch: image 0 is {orig_width}x{orig_height}, "
                    f"but image {i} is {img.size[0]}x{img.size[1]}. "
                    f"All images in a batch must have identical dimensions."
                )

        target_width = ((orig_width + _ROUND_TO - 1) // _ROUND_TO) * _ROUND_TO
        target_height = ((orig_height + _ROUND_TO - 1) // _ROUND_TO) * _ROUND_TO

        print(f"Batch dimensions: {orig_width}x{orig_height}, {batch_size} images")
        if target_width != orig_width or target_height != orig_height:
            print(f"Rounded to: {target_width}x{target_height} (multiple of {_ROUND_TO})")
        print(f"Steps: {gen_config['num_inference_steps']}, Guidance Scale: {gen_config['guidance_scale']}")

        # 5. Prepare generator
        generator = None
        if gen_config['generator_seed'] is not None:
            generator = torch.Generator(device=self.device).manual_seed(gen_config['generator_seed'])

        # 6. Progress callback (disabled when logging to file)
        progress_bar = tqdm(total=gen_config['num_inference_steps'], desc="Generating batch", unit="step", disable=not sys.stdout.isatty())

        def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
            progress_bar.update(1)
            return callback_kwargs

        # 7. Conditioning scale
        conditioning_scale = gen_config['controlnet_conditioning_scale']

        # 8. TRUE BATCH PROCESSING: Single forward pass for all images
        diffusion_start = time.time()
        result = self.pipe(
            prompt=full_prompts,
            negative_prompt=full_negative_prompts,
            image=images_pil,
            mask_image=masks_pil,
            control_image=control_images,
            height=target_height,
            width=target_width,
            num_inference_steps=gen_config['num_inference_steps'],
            guidance_scale=gen_config['guidance_scale'],
            controlnet_conditioning_scale=conditioning_scale,
            num_images_per_prompt=1,
            strength=gen_config['strength'],
            generator=generator,
            callback_on_step_end=callback_on_step_end,
        )
        diffusion_time = time.time() - diffusion_start

        progress_bar.close()
        print(f"[BATCH] ✓ Batch complete in {diffusion_time:.1f}s ({diffusion_time/batch_size:.2f}s per image)")

        # 9. Convert results to numpy arrays
        results = [np.array(img) for img in result.images]

        return results


