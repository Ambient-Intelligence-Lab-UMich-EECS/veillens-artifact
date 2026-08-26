#!/usr/bin/python
"""
SDXL Pipeline Adapter

Wraps the core SDXL inpainting pipeline for use in the main geolocation pipeline.
Provides a simplified interface for processing individual clues.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from veillens.edit.pipeline import SDXLInpaintingPipeline
from veillens.paths import FINAL_DIR, INTERMEDIATE_DIR, SEGMENTATION_DIR
from veillens.privacy.spec import get_spec
from veillens.reason.oracle import GeoLocationAPI


class SDXLPipelineAdapter:
    """
    Adapter for the core SDXL inpainting pipeline.
    Simplifies the interface for clue-by-clue processing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with an already-loaded configuration.

        Args:
            config: config dict from ``veillens.config.load_config``, with ``${paths.*}``
                placeholders already resolved.
        """
        self.pipeline = SDXLInpaintingPipeline(config)

        print(f"[Adapter] Initialized SDXL pipeline with config: {config.get('_config_path', '<in-memory>')}")

        # Store config reference for easy access
        self.config = self.pipeline.config
        self.spec = get_spec(self.config)

    def _calculate_sliding_window_regions(
        self,
        w: int,
        h: int,
        crop_w: int,
        crop_h: int,
        overlap: float
    ) -> List[Tuple[str, int, int, int, int]]:
        """
        Calculate all region coordinates using sliding window approach.

        Edge cases:
        - Image smaller than crop: Use full image as single region (no padding)
        - Boundary crops: Include all, even if overlap exceeds specified percentage

        Args:
            w: Image width
            h: Image height
            crop_w: Crop width
            crop_h: Crop height
            overlap: Overlap percentage (0.0 to 1.0)

        Returns:
            List of (region_name, x1, y1, x2, y2) tuples
        """
        regions = []

        # Handle image smaller than crop size - skip R2 entirely
        # R2 would just re-evaluate the full image which R1 already did
        if w <= crop_w and h <= crop_h:
            return regions  # Return empty list, no R2 regions needed

        stride_x = int(crop_w * (1 - overlap))
        stride_y = int(crop_h * (1 - overlap))

        # Ensure stride is at least 1 to avoid infinite loop
        stride_x = max(1, stride_x)
        stride_y = max(1, stride_y)

        region_idx = 0
        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate crop boundaries
                x1 = x
                y1 = y
                x2 = min(x + crop_w, w)
                y2 = min(y + crop_h, h)

                # Align to boundary if crop would exceed - ensure full crop size
                if x2 == w and x2 - x1 < crop_w:
                    x1 = max(0, w - crop_w)
                if y2 == h and y2 - y1 < crop_h:
                    y1 = max(0, h - crop_h)

                regions.append((str(region_idx), x1, y1, x2, y2))
                region_idx += 1

                # Move to next position or break if at boundary
                if x2 >= w:
                    break
                x += stride_x

            if y2 >= h:
                break
            y += stride_y

        return regions

    def _upscale_region_proportional(
        self,
        region: np.ndarray,
        upscale_target: int
    ) -> np.ndarray:
        """
        Upscale region proportionally based on aspect ratio.

        The longer dimension scales to upscale_target, shorter scales proportionally.
        Example: 768x768 -> 2048x2048, 768x512 -> 2048x1365

        Args:
            region: Region image array (H, W, C)
            upscale_target: Target size for longer dimension

        Returns:
            Upscaled region array
        """
        h, w = region.shape[:2]
        if w >= h:
            new_w = upscale_target
            new_h = int(h * (upscale_target / w))
        else:
            new_h = upscale_target
            new_w = int(w * (upscale_target / h))
        return cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def process_clue(
        self,
        image_path: str,
        image_id: str,
        object_description: str,
        target_description: str,
        negative_prompt: str = None
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Process a single clue: detect, segment, and inpaint.

        Args:
            image_path: Path to input image
            image_id: Image identifier (for file naming)
            object_description: Description of object to segment (for detection)
            target_description: Description of what to generate (for inpainting)
            negative_prompt: Optional negative prompt

        Returns:
            Tuple of (mask_path, mask_array, inpainted_image_array)
        """
        print(f"\n[Adapter] Processing clue: {object_description[:50]}...")

        # Use default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = self.config['generation']['default_negative_prompt']

        # Run the core pipeline
        results = self.pipeline.run(
            image_path=image_path,
            object_description=object_description,
            target_description=target_description,
            negative_prompt=negative_prompt,
            image_id_override=image_id,
            region_tag="across region"
        )

        # Load the generated mask and final image
        mask_path = results['paths']['mask']
        final_path = results['paths']['final']

        # Load as numpy arrays for compositing
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        final_image = cv2.imread(final_path)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        print(f"[Adapter] Generated mask: {mask_path}")
        print(f"[Adapter] Generated result: {final_path}")

        return mask_path, mask, final_image

    def composite_result(
        self,
        base_image: np.ndarray,
        inpainted_image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Composite the inpainted region onto the base image using the mask.

        Args:
            base_image: Base image (RGB)
            inpainted_image: Inpainted image (RGB)
            mask: Binary mask (grayscale, 0-255)

        Returns:
            Composited image (RGB)
        """
        # Normalize mask to [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0

        # Expand mask to 3 channels
        mask_3ch = np.stack([mask_norm] * 3, axis=-1)

        # Composite: result = base * (1 - mask) + inpainted * mask
        result = base_image * (1 - mask_3ch) + inpainted_image * mask_3ch

        return result.astype(np.uint8)

    def save_final_image(self, image: np.ndarray, image_id: str) -> str:
        """
        Save the final processed image.

        Args:
            image: Image to save (RGB)
            image_id: Image identifier

        Returns:
            Path to saved image
        """
        # Get output directories from the pipeline config
        output_base = Path(self.pipeline.config['output']['base_dir'])
        final_dir = output_base / FINAL_DIR
        final_dir.mkdir(parents=True, exist_ok=True)

        final_path = final_dir / f"{image_id}_final.png"

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(final_path), image_bgr)

        print(f"[Adapter] Saved final image: {final_path}")

        return str(final_path)

    def extract_regions(self, image_path: str = None) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract regions from the image using sliding window approach and upscale proportionally.

        Uses configurable sliding window parameters from round2_regions config:
        - crop_width: Width of each crop
        - crop_height: Height of each crop
        - overlap_percent: Overlap between adjacent crops
        - upscale_target: Target size for longer dimension (proportional scaling)

        Args:
            image_path: Path to the image. If None, uses self.image_path.

        Returns:
            List of (region_name, upscaled_region_array, (x1, y1, x2, y2)) tuples
            Region names are integer strings ("0", "1", "2", ...)
        """
        # Use provided image path or default to self.image_path
        source_path = image_path if image_path is not None else self.image_path

        # Load image
        image = cv2.imread(source_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Get sliding window config parameters
        r2_config = self.config.get('round2_regions', {})
        crop_w = r2_config.get('crop_width', 768)
        crop_h = r2_config.get('crop_height', 768)
        overlap = r2_config.get('overlap_percent', 0.1)
        upscale_target = r2_config.get('upscale_target', 2048)

        print(f"[Adapter] Extracting regions from image ({w}x{h}) with sliding window")
        print(f"[Adapter] Crop size: {crop_w}x{crop_h}, overlap: {overlap*100:.0f}%, upscale target: {upscale_target}")

        # Calculate region coordinates using sliding window
        regions = self._calculate_sliding_window_regions(w, h, crop_w, crop_h, overlap)

        print(f"[Adapter] Generated {len(regions)} regions")

        extracted_regions = []
        for region_name, x1, y1, x2, y2 in regions:
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            # Extract region
            region = image[y1:y2, x1:x2]

            # Upscale proportionally
            upscaled_region = self._upscale_region_proportional(region, upscale_target)

            extracted_regions.append((region_name, upscaled_region, (x1, y1, x2, y2)))
            upscaled_h, upscaled_w = upscaled_region.shape[:2]
            print(f"[Adapter] Extracted region '{region_name}': {x1},{y1} -> {x2},{y2}, upscaled to {upscaled_w}x{upscaled_h}")

        return extracted_regions

    def process_clue_with_region_mapping(
        self,
        image_path: str,
        image_id: str,
        object_description: str,
        target_description: str,
        region_tag: str,
        crop_coords: Tuple[int, int, int, int] = None,
        negative_prompt: str = None
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Process a single clue with region mapping for segmentation.

        Args:
            image_path: Path to full input image
            image_id: Image identifier
            object_description: Description of object to segment
            target_description: Description of what to generate
            region_tag: Region tag ("across region" or specific region)
            crop_coords: If region is specific, (x1, y1, x2, y2) crop coordinates
            negative_prompt: Optional negative prompt

        Returns:
            Tuple of (mask_path, mask_array, inpainted_image_array)
        """
        print(f"\n[Adapter] Processing clue with region mapping: {object_description[:50]}...")
        print(f"[Adapter] Region: {region_tag}")

        if region_tag == "across region":
            # Process full image (existing logic)
            return self.process_clue(
                image_path=image_path,
                image_id=image_id,
                object_description=object_description,
                target_description=target_description,
                negative_prompt=negative_prompt
            )
        else:
            # Process with region mapping
            if crop_coords is None:
                raise ValueError(f"crop_coords required for region '{region_tag}'")

            x1, y1, x2, y2 = crop_coords

            # Load full image
            full_image = cv2.imread(image_path)
            full_image_rgb = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

            # Crop region
            region = full_image_rgb[y1:y2, x1:x2]

            # Upscale proportionally
            upscale_target = self.config.get('round2_regions', {}).get('upscale_target', 2048)
            upscaled_region = self._upscale_region_proportional(region, upscale_target)

            # Save upscaled region temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                upscaled_region_bgr = cv2.cvtColor(upscaled_region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, upscaled_region_bgr)

            try:
                # Run Grounded SAM 2 on upscaled region
                results = self.pipeline.run(
                    image_path=temp_path,
                    object_description=object_description,
                    target_description=target_description,
                    negative_prompt=negative_prompt or self.pipeline.config['generation']['default_negative_prompt'],
                    image_id_override=image_id,
                    region_tag=region_tag
                )

                # Load mask from results
                mask_path_temp = results['paths']['mask']
                mask_upscaled = cv2.imread(mask_path_temp, cv2.IMREAD_GRAYSCALE)

                # Downscale mask back to original crop size
                original_crop_h = y2 - y1
                original_crop_w = x2 - x1
                mask_downscaled = cv2.resize(mask_upscaled, (original_crop_w, original_crop_h), interpolation=cv2.INTER_NEAREST)

                # Create full-image mask and place downscaled mask at crop location
                h, w = full_image_rgb.shape[:2]
                mask_full = np.zeros((h, w), dtype=np.uint8)
                mask_full[y1:y2, x1:x2] = mask_downscaled

                # Load inpainted result and downscale
                final_path_temp = results['paths']['final']
                inpainted_upscaled = cv2.imread(final_path_temp)
                inpainted_upscaled_rgb = cv2.cvtColor(inpainted_upscaled, cv2.COLOR_BGR2RGB)
                inpainted_downscaled = cv2.resize(inpainted_upscaled_rgb, (original_crop_w, original_crop_h), interpolation=cv2.INTER_LANCZOS4)

                # Create full inpainted image
                inpainted_full = full_image_rgb.copy()
                inpainted_full[y1:y2, x1:x2] = inpainted_downscaled

                # Save final mask in intermediate directory with region subdirectory
                output_base = Path(self.pipeline.config['output']['base_dir'])
                segmentation_base = output_base / INTERMEDIATE_DIR / image_id / SEGMENTATION_DIR

                # Create region-specific subdirectory
                region_subdir = region_tag.replace(' ', '_')
                segmentation_dir = segmentation_base / region_subdir
                segmentation_dir.mkdir(parents=True, exist_ok=True)

                # Create feature name from object description (first 10 chars, sanitized)
                feature_name = object_description[:10].replace(' ', '_').replace('.', '').replace('/', '_').replace('\\', '_')

                # Use unique filename with feature name to prevent overwrites
                mask_filename = f"{image_id}_{region_tag.replace(' ', '_')}_{feature_name}_resize_mask.png"
                mask_path = str(segmentation_dir / mask_filename)
                cv2.imwrite(mask_path, mask_full)

                print(f"[Adapter] Generated mask with region mapping: {mask_path}")

                return mask_path, mask_full, inpainted_full

            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

    async def evaluate_full_image(
        self,
        image_path: str,
        geo_api: GeoLocationAPI,
        cache_dir: Path
    ) -> Dict[str, Any]:
        """
        Round 1: Evaluate full image only for across-region clues.

        Args:
            image_path: Path to the image
            geo_api: GeoLocationAPI instance
            cache_dir: Path to original_cache directory

        Returns:
            Dict with full image evaluation result and cache path
        """
        print(f"\n[Adapter] Round 1: Full image evaluation for {image_path}")

        # Create cache directory for this image
        image_name = Path(image_path).stem
        image_cache_dir = cache_dir / image_name
        image_cache_dir.mkdir(parents=True, exist_ok=True)

        # Full image analysis
        result = await geo_api.get_geolocation_prediction(
            image_path=image_path,
            schema_type='original',
            role='reason'
        )

        if not result:
            print("[Adapter] Round 1 evaluation failed")
            return None

        # Tag all features as "across region"
        for feature in result['critical_features']['features']:
            feature['region'] = 'across region'

        # Save result as round1_full_img.json
        cache_file = image_cache_dir / "round1_full_img.json"
        with open(cache_file, 'w') as f:
            import json
            json.dump(result, f, indent=2)

        num_features = len(result['critical_features']['features'])
        print(f"[Adapter] Round 1: {num_features} across-region clues found")
        print(f"[Adapter] Saved Round 1 result: {cache_file}")

        # Build Round 1 context string for Round 2 (opaque to downstream callers)
        predicted_location = self.spec.format_round1_context(result)

        return {
            'result': result,
            'cache_path': str(cache_file),
            'predicted_location': predicted_location,
            'image_name': image_name
        }

    async def evaluate_regional_crops(
        self,
        processed_image_path: str,
        predicted_location: str,
        geo_api: GeoLocationAPI,
        cache_dir: Path,
        original_image_name: str
    ) -> Dict[str, Any]:
        """
        Round 2: Evaluate regional crops from processed image using sliding window.

        Args:
            processed_image_path: Path to Round 1 processed image
            predicted_location: Predicted location from Round 1
            geo_api: GeoLocationAPI instance
            cache_dir: Path to original_cache directory
            original_image_name: Original image name for cache directory

        Returns:
            Dict with regional results and cache paths
        """
        print(f"\n[Adapter] Round 2: Regional crop evaluation for {processed_image_path}")

        # Get cache directory
        image_cache_dir = cache_dir / original_image_name

        # Extract regions from processed image
        regions = self.extract_regions(processed_image_path)

        # Helper function to process a single region
        async def process_region(region_name, upscaled_region, crop_coords):
            region_start = time.time()

            # Save upscaled region temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                upscaled_region_bgr = cv2.cvtColor(upscaled_region, cv2.COLOR_RGB2BGR)
                cv2.imwrite(temp_path, upscaled_region_bgr)

                # Force filesystem flush to prevent race condition in concurrent execution
                with open(temp_path, 'r+b') as f:
                    f.flush()
                    os.fsync(f.fileno())

            try:
                # Get prediction for this region
                region_result = await geo_api.get_geolocation_prediction(
                    image_path=temp_path,
                    schema_type='original',
                    predicted_location=predicted_location,
                    region_name=region_name,
                    cache_key_override=original_image_name,
                    role='reason'
                )

                if region_result:
                    # Tag features with region name
                    for feature in region_result['critical_features']['features']:
                        feature['region'] = region_name

                    num_features = len(region_result['critical_features']['features'])
                    print(f"[Adapter] Region '{region_name}': {num_features} clues found")

                    # Save individual region result as round2_{region}.json
                    region_filename = 'round2_' + region_name.replace(' ', '_') + '.json'
                    region_cache_file = image_cache_dir / region_filename
                    with open(region_cache_file, 'w') as f:
                        import json
                        json.dump(region_result, f, indent=2)
                    print(f"[Adapter] Saved region result: {region_cache_file}")

                    # Track regional timing

                    return region_name, region_result, str(region_cache_file)
                else:
                    print(f"[Adapter] Region '{region_name}': evaluation failed")
                    return region_name, None, None

            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

        # Process all regions concurrently
        tasks = [process_region(name, region, coords) for name, region, coords in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        regional_results = []
        cache_paths = []
        for region_name, region_result, cache_path in results:
            # Handle exceptions gracefully
            if isinstance(region_result, Exception):
                print(f"[Adapter] Region '{region_name}' failed with exception: {region_result}")
                continue

            if region_result:
                regional_results.append(region_result)
                cache_paths.append(cache_path)

        total_features = sum(len(r['critical_features']['features']) for r in regional_results)
        print(f"[Adapter] Round 2: {total_features} total regional clues found")

        return {
            'results': regional_results,
            'cache_paths': cache_paths
        }


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Import it in your main pipeline.")
