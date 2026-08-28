"""
Stage 2 (grounding), detection + segmentation half.

`GroundedSAM2Segmenter` turns a language cue into pixel masks: Grounding DINO proposes boxes for
an open-set text prompt, they are filtered against the cue's target concept, and SAM 2 turns the
surviving boxes into masks.

Grounding DINO scores every text token against every box, and the box with the highest overall
score is often not the one that matches the cue: a prompt like "campaign poster on a fence" will
happily return the fence. `select_boxes_by_target_concept` aligns the target concept's tokens to
their positions in the caption and re-ranks on those columns alone, which is why the segmenter
carries its own `predict_with_token_scores` instead of calling upstream's `predict`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from veillens.paths import bootstrap_grounded_sam2

# Upstream Grounded-SAM-2 is not an installable package, so its root must be on sys.path before
# `sam2` / `grounding_dino` resolve.
bootstrap_grounded_sam2()

from grounding_dino.groundingdino.util.inference import (  # noqa: E402
    load_image,
    load_model,
)
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from torchvision.ops import box_convert  # noqa: E402

logger = logging.getLogger(__name__)


class GroundedSAM2Segmenter:
    """Handles object detection and segmentation using Grounded-SAM-2"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['detection']['device']
        self.logger = logging.getLogger(__name__)

        logging.getLogger('root').setLevel(logging.INFO)

        # Get Grounded-SAM-2 root (path already added at module level)
        grounded_sam2_root = Path(config['models']['grounded_sam2_root'])

        # Enable float16 precision BEFORE loading models
        if self.device == "cuda":
            print("Using float16 precision for Grounding DINO compatibility")
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

            # TF32 needs an Ampere-or-newer GPU; on anything older the flags are ignored.
            if (config['performance']['use_tf32']
                    and torch.cuda.get_device_properties(0).major >= 8):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TF32 enabled for matmul and cuDNN")

        # Build SAM2 model
        # SAM2's build_sam2 expects a relative config name (uses Hydra)
        # It looks for configs relative to the sam2 package
        sam2_checkpoint = config['models']['sam2_checkpoint']
        sam2_config = config['models']['sam2_model_config']  # Keep as relative path

        print(f"\nLoading SAM2 model from {sam2_checkpoint}...")
        print(f"Using SAM2 config: {sam2_config}")
        print("      (This may take 1-3 minutes on networked storage...)")
        sam2_start = time.time()
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)

        # Convert SAM2 to float16 for optimal performance
        if self.device == "cuda":
            self.sam2_model = self.sam2_model.half()
            print(f"      ✓ SAM2 loaded and converted to float16 in {time.time() - sam2_start:.1f}s")
        else:
            print(f"      ✓ SAM2 model loaded in {time.time() - sam2_start:.1f}s")

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Build Grounding DINO model (inside autocast context)
        grounding_config = grounded_sam2_root / config['models']['grounding_dino_config']
        grounding_checkpoint = config['models']['grounding_dino_checkpoint']

        print(f"\nLoading Grounding DINO model from {grounding_checkpoint}...")
        print("      (This may take 1-2 minutes on networked storage...)")
        dino_start = time.time()
        self.grounding_model = load_model(
            model_config_path=str(grounding_config),
            model_checkpoint_path=grounding_checkpoint,
            device=self.device
        )
        dino_time = time.time() - dino_start
        print(f"      ✓ Grounding DINO model loaded in {dino_time:.1f}s")

    def predict_with_token_scores(
        self,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Enhanced predict that returns full token-level logits.

        Args:
            image: Preprocessed image tensor
            caption: Text description for detection
            box_threshold: Minimum box confidence threshold
            text_threshold: Minimum text alignment threshold

        Returns:
            boxes: (n, 4) bounding boxes in cxcywh format
            confidences: (n,) max confidence per box
            token_logits: (n, 256) FULL token scores per box
            labels: List[str] decoded phrases
        """
        from grounding_dino.groundingdino.util.inference import (
            get_phrases_from_posmap,
            preprocess_caption,
        )

        caption = preprocess_caption(caption=caption)
        model = self.grounding_model.to(self.device)
        image = image.to(self.device)

        # Model forward pass
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])

        # Extract logits and boxes
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (900, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]              # (900, 4)

        # Filter boxes by confidence (Stage 1)
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        token_logits = prediction_logits[mask]  # (n, 256) - KEEP FULL LOGITS
        boxes = prediction_boxes[mask]          # (n, 4)

        # Tokenize caption for phrase extraction
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        # Extract phrases (same as original predict())
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in token_logits
        ]

        # Return max confidence + full logits
        confidences = token_logits.max(dim=1)[0]

        return boxes, confidences, token_logits, phrases

    def align_target_concept_tokens(
        self,
        target_concept: str,
        full_caption: str
    ) -> List[int]:
        """
        Find token positions of target_concept within full caption tokenization.
        Handles subword tokenization (e.g., "rectangular" → ["rect", "##angular"]).

        Args:
            target_concept: Target phrase (e.g., "street sign")
            full_caption: Full description (e.g., "blue rectangular street sign.")

        Returns:
            List of token positions in full caption's tokenization
            (positions are offset by 1 to skip [CLS] token)
        """
        tokenizer = self.grounding_model.tokenizer

        # Tokenize both
        target_tokens = tokenizer(target_concept.lower() + ".")['input_ids']
        full_tokens = tokenizer(full_caption.lower() + ".")['input_ids']

        # Remove special tokens from target
        # [CLS]=101, [SEP]=102, .=1012
        target_clean = [t for t in target_tokens if t not in [101, 102, 1012]]

        # Find subsequence in full tokenization
        positions = []
        for i in range(len(full_tokens) - len(target_clean) + 1):
            if full_tokens[i:i+len(target_clean)] == target_clean:
                # Found exact match - return these positions
                positions = list(range(i, i + len(target_clean)))
                break

        # Fallback: Individual token matching if no exact subsequence
        if not positions:
            for i, token in enumerate(full_tokens):
                if token in target_clean:
                    positions.append(i)

        # Filter out [CLS] position (always 0)
        positions = [p for p in positions if p > 0]

        # Adjust for 256-token logits array (which includes [CLS] at position 0)
        # No adjustment needed - positions already 0-indexed

        return positions

    def select_boxes_by_target_concept(
        self,
        boxes: torch.Tensor,
        token_logits: torch.Tensor,
        target_concept: str,
        full_caption: str,
        text_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Three-stage filtering using target concept token scores.

        Args:
            boxes: (n, 4) boxes from Stage 1 (already filtered by box_threshold)
            token_logits: (n, 256) full token scores
            target_concept: Target phrase (e.g., "street sign")
            full_caption: Full description
            text_threshold: Minimum text score threshold

        Returns:
            filtered_boxes: (k, 4) top-k boxes
            filtered_logits: (k, 256) corresponding logits
            selected_indices: List of original indices (for tracking)
        """
        # Get target token positions
        target_positions = self.align_target_concept_tokens(target_concept, full_caption)

        if not target_positions:
            print(f"⚠️  Could not align target_concept '{target_concept}' - using all tokens")
            target_positions = list(range(1, 256))  # Fallback: all non-[CLS] tokens

        self.logger.debug(f"Target concept '{target_concept}' → token positions: {target_positions}")

        # Extract target token scores: (n, num_target_tokens)
        target_scores = token_logits[:, target_positions]

        # Stage 2: Filter by MAX target token score > text_threshold
        max_target_scores = target_scores.max(dim=1)[0]  # (n,)
        stage2_mask = max_target_scores > text_threshold

        if stage2_mask.sum() == 0:
            print(f"⚠️  No boxes passed Stage 2 (max target score > {text_threshold})")
            print(f"    Keeping all {len(boxes)} boxes from Stage 1")
            stage2_mask = torch.ones(len(boxes), dtype=torch.bool)

        # Apply Stage 2 filter
        boxes_s2 = boxes[stage2_mask]
        logits_s2 = token_logits[stage2_mask]
        target_scores_s2 = target_scores[stage2_mask]
        indices_s2 = torch.arange(len(boxes))[stage2_mask]

        self.logger.debug(f"Stage 2: {stage2_mask.sum()}/{len(boxes)} boxes with max target score > {text_threshold}")

        # Stage 3: Select top-k by AVERAGE target token score
        avg_target_scores = target_scores_s2.mean(dim=1)  # (m,)

        top_k = self.config['detection'].get('top_k_boxes', 3)
        k = min(top_k, len(boxes_s2))

        top_k_indices = torch.topk(avg_target_scores, k).indices

        # Select top-k boxes
        final_boxes = boxes_s2[top_k_indices]
        final_logits = logits_s2[top_k_indices]
        final_indices = indices_s2[top_k_indices].tolist()

        self.logger.debug(f"Stage 3: Selected top-{k} boxes by avg target score")

        return final_boxes, final_logits, final_indices

    def segment(
        self,
        image_path: str,
        object_description: str,
        target_concept: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect and segment objects in the image.

        Args:
            image_path: Path to input image
            object_description: Text description of object to segment (e.g., "car. tire.")
            target_concept: Target concept for detection (e.g., "street sign"). Optional.

        Returns:
            Tuple of (image_source, masks, boxes, labels)
        """
        # Load image
        image_source, image = load_image(image_path)

        # Ensure text prompt ends with a dot
        text = object_description.lower()
        if not text.endswith('.'):
            text += '.'

        # Detect objects with Grounding DINO
        self.logger.info(f"Detecting objects: {text}")
        boxes, confidences, token_logits, labels = self.predict_with_token_scores(
            image=image,
            caption=text,
            box_threshold=self.config['detection']['box_threshold'],
            text_threshold=self.config['detection']['text_threshold']
        )

        # A clue whose object cannot be detected above threshold is left unmasked; the caller
        # handles the empty result with a len(masks) == 0 check.
        if len(boxes) == 0:
            print(f"⚠️  No boxes above threshold ({self.config['detection']['box_threshold']}) - skipping this feature")
            return image_source, np.array([]), np.array([]), []
        else:
            # Apply target concept filtering if provided
            if target_concept:
                boxes, token_logits, selected_indices = self.select_boxes_by_target_concept(
                    boxes=boxes,
                    token_logits=token_logits,
                    target_concept=target_concept,
                    full_caption=text,
                    text_threshold=self.config['detection']['text_threshold']
                )
                # Update labels to match selected boxes
                labels = [labels[i] for i in selected_indices]
                self.logger.debug(f"After target filtering: {len(boxes)} box(es)")

        # Convert boxes for SAM2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Segment with SAM2
        self.sam2_predictor.set_image(image_source)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert to (n, H, W) format
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Log completion
        if len(masks) > 0:
            self.logger.info(f"✓ Segmented {len(masks)} object(s)")

        return image_source, masks, input_boxes, labels

    @staticmethod
    def combine_masks(masks: np.ndarray) -> np.ndarray:
        """Combine multiple masks into a single binary mask"""
        if masks.ndim == 3:
            combined = np.any(masks, axis=0).astype(np.uint8) * 255
        else:
            combined = (masks > 0).astype(np.uint8) * 255
        return combined


