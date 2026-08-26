"""
Stage 3 (editing), façade.

`SDXLInpaintingPipeline` wires the segmenter, the conditioning extractor and the inpainter into
one object. It is the single-image convenience path; the batch loop in `veillens.loop` drives the
same three components directly.

Extracted verbatim from the original `core/sdxl_pipeline.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from veillens.edit.control import DepthExtractor
from veillens.edit.inpaint import ControlNetInpainter
from veillens.ground.segmenter import GroundedSAM2Segmenter
from veillens.paths import FINAL_DIR, INTERMEDIATE_DIR, SEGMENTATION_DIR


class SDXLInpaintingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: an already-loaded config dict, as returned by
                ``veillens.config.load_config``. It must be the loaded dict rather than a path:
                loading it here would skip ``substitute_paths``, and every ``${paths.*}`` in the
                model section would reach torch as a literal string.
        """
        self.config = config

        # Initialize components
        print("Initializing Grounded-SAM-2 Segmenter...")
        self.segmenter = GroundedSAM2Segmenter(self.config)

        print("Initializing Edge and Depth Extractor...")
        self.extractor = DepthExtractor(self.config)

        print("Initializing ControlNet Inpainter...")
        self.inpainter = ControlNetInpainter(self.config)

        # Clear GPU cache to free fragmented memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ GPU cache cleared")

        print("Pipeline initialization complete!\n")

    def run(
        self,
        image_path: str,
        object_description: str,
        target_description: str,
        negative_prompt: Optional[str] = None,
        image_id_override: str = None,
        region_tag: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete inpainting pipeline.

        Args:
            image_path: Path to input image
            object_description: Description of object to segment
            target_description: Description of what to generate in the masked area
            negative_prompt: Negative prompt to avoid unwanted features (optional)
            image_id_override: Override image ID (used when image_path is temp file)
            region_tag: Region tag for organizing segmentation results (e.g., "top left", "across region")

        Returns:
            Dictionary containing all results and paths
        """
        # Setup output directory: intermediate/{image_name}/
        image_name = image_id_override if image_id_override else Path(image_path).stem
        base_dir = Path(self.config['output']['base_dir'])
        output_base = base_dir / INTERMEDIATE_DIR / image_name
        output_base.mkdir(parents=True, exist_ok=True)

        results = {'paths': {}}

        # Create subdirectories for segmentation and final result within the image folder
        (output_base / SEGMENTATION_DIR).mkdir(exist_ok=True)
        (output_base / FINAL_DIR).mkdir(exist_ok=True)

        print("=" * 60)
        print("STEP 1: Object Detection and Segmentation")
        print("=" * 60)
        image_source, masks, boxes, labels = self.segmenter.segment(
            image_path, object_description
        )

        # Combine masks and save with custom filename
        combined_mask = self.segmenter.combine_masks(masks)

        # Create filename: image_name + first 10 chars of object description
        obj_desc_short = object_description[:10].replace(' ', '_').replace('.', '')
        segmentation_filename = f"{image_name}_{obj_desc_short}"

        # Create region-specific subdirectory for segmentation results
        segmentation_base = output_base / SEGMENTATION_DIR
        if region_tag and region_tag != "across region":
            # Create region subdirectory: segmentation/top_left/, etc.
            region_subdir = region_tag.replace(' ', '_')
            segmentation_dir = segmentation_base / region_subdir
            segmentation_dir.mkdir(parents=True, exist_ok=True)
        elif region_tag == "across region":
            # Create full_img subdirectory for across-region clues
            segmentation_dir = segmentation_base / "full_img"
            segmentation_dir.mkdir(parents=True, exist_ok=True)
        else:
            # No region tag: save to flat segmentation/ (backward compatibility)
            segmentation_dir = segmentation_base
            segmentation_dir.mkdir(exist_ok=True)

        # Save binary mask in region-specific subdirectory
        mask_path = segmentation_dir / f'{segmentation_filename}_mask.png'
        cv2.imwrite(str(mask_path), combined_mask)
        results['paths']['mask'] = str(mask_path)

        # Save mask overlay in region-specific subdirectory
        overlay = image_source.copy()
        overlay[combined_mask > 0] = overlay[combined_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay_path = segmentation_dir / f'{segmentation_filename}_overlay.jpg'
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        results['paths']['mask_overlay'] = str(overlay_path)
        print(f"Saved segmentation mask to {mask_path}")
        print(f"Saved segmentation overlay to {overlay_path}")

        print("\n" + "=" * 60)
        print("STEP 2: Extract Depth Map")
        print("=" * 60)
        depth_map = self.extractor.extract_depth(image_source)
        print("Depth map extracted (not cached)")

        print("\n" + "=" * 60)
        print("STEP 3: Inpainting with ControlNet")
        print("=" * 60)

        # Convert to PIL Images for diffusers
        # Ensure RGB format for image
        image_pil = Image.fromarray(image_source).convert('RGB')
        # Mask as grayscale (white=inpaint, black=keep)
        mask_pil = Image.fromarray(combined_mask).convert('L')
        # Depth: Convert grayscale depth to RGB (NOT colorized - ControlNet expects grayscale values)
        depth_pil = Image.fromarray(depth_map).convert('L').convert('RGB')

        # Prepare negative prompt
        if negative_prompt is None:
            negative_prompt = self.config['generation'].get(
                'default_negative_prompt',
                "blurry, ugly, distorted, low quality, bad anatomy, watermark"
            )

        # Perform inpainting
        result_image = self.inpainter.inpaint(
            image=image_pil,
            mask=mask_pil,
            depth_map=depth_pil,
            prompt=target_description,
            negative_prompt=negative_prompt
        )

        # Save final result
        final_path = output_base / FINAL_DIR / 'inpainted.jpg'
        result_image.save(str(final_path), quality=95)
        results['paths']['final'] = str(final_path)
        print(f"Saved final inpainted image to {final_path}")

        # Save side-by-side comparison
        comparison = Image.new('RGB', (image_pil.width * 2, image_pil.height))
        comparison.paste(image_pil, (0, 0))
        comparison.paste(result_image, (image_pil.width, 0))
        comparison_path = output_base / FINAL_DIR / 'comparison.jpg'
        comparison.save(str(comparison_path), quality=95)
        results['paths']['comparison'] = str(comparison_path)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults saved to: {output_base}")
        print(f"Final inpainted image: {final_path}")

        return results


