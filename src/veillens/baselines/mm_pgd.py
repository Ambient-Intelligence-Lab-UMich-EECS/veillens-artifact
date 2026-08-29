"""
`mm-pgd` — Union-CAM-masked PGD against a Places365 scene-classifier ensemble.

Two pieces, both from the multi-model attack literature:

* **Union-CAM.** One Grad-CAM per model for its own locked top-1 class, blurred, normalised, then
  combined by pixelwise maximum. Thresholded at the 20th percentile from the top and softly
  rescaled, it becomes a [0, 1] mask.
* **Masked PGD.** Standard sign-gradient ascent on the ensemble cross-entropy toward each model's
  *own clean prediction* — so the step moves away from what the model already believes — with the
  gradient multiplied by the mask before the sign is taken, and the perturbation projected back
  into the epsilon-ball in [0, 1] pixel space at every step.

Weights: `resnet18_places365.pth.tar` and `resnet50_places365.pth.tar`
(`scripts/fetch_baseline_weights.py`). The checkpoints load into stock torchvision ResNets
unchanged, so no architecture is vendored here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from veillens.baselines.base import Baseline

__all__ = ["MultiModelPGD"]

#: ImageNet statistics, which is what the Places365 checkpoints were trained under.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _stats_tensors(device, dtype):
    import torch
    mean = torch.tensor(_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def _load_places365(arch: str, weight_path: str, device: str):
    """A torchvision ResNet carrying Places365 weights, in eval mode on *device*."""
    import torch
    import torchvision

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # The release was saved from a DataParallel wrapper, so every key carries a 'module.' prefix.
    state = {key.replace("module.", ""): value for key, value in state.items()}

    model = getattr(torchvision.models, arch)(num_classes=365)
    model.load_state_dict(state)
    return model.to(device).eval()


class MultiModelPGD(Baseline):
    name = "mm-pgd"
    summary = "Union-CAM-masked PGD against a Places365 ResNet-18 + ResNet-50 ensemble"

    def __init__(self, models: List[Dict[str, str]] | None = None,
                 epsilon: float = 8 / 255, num_steps: int = 20, alpha: float | None = None,
                 cam_percentile: float = 20.0, cam_target_layer: str = "layer4",
                 cam_blur_sigma: float = 1.0, use_amp: bool = True,
                 device: str = "cuda", **params: Any) -> None:
        super().__init__(models=models, epsilon=epsilon, num_steps=num_steps, alpha=alpha,
                         cam_percentile=cam_percentile, cam_target_layer=cam_target_layer,
                         cam_blur_sigma=cam_blur_sigma, use_amp=use_amp, device=device, **params)
        self.model_specs = models or []
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha if alpha is not None else epsilon / num_steps
        self.cam_percentile = cam_percentile
        self.cam_target_layer = cam_target_layer
        self.cam_blur_sigma = cam_blur_sigma
        self.use_amp = use_amp
        self.device = device
        self.models: List[Any] = []
        self.cams: List[Any] = []

    @classmethod
    def check_prerequisites(cls, params: Dict[str, Any]) -> List[str]:
        problems = []
        for spec in params.get("models") or []:
            weight = spec.get("weight_path")
            if not weight or not Path(weight).is_file():
                problems.append(
                    f"missing Places365 checkpoint: {weight or '<unset>'}\n"
                    "      python scripts/fetch_baseline_weights.py --method mm-pgd"
                )
        if not params.get("models"):
            problems.append("mm-pgd has no models configured; see configs/baselines.yaml")
        return problems

    def prepare(self, items: List[Dict[str, Any]]) -> None:
        from pytorch_grad_cam import GradCAM

        self.models = [_load_places365(spec["arch"], spec["weight_path"], self.device)
                       for spec in self.model_specs]
        self.cams = [GradCAM(model=model,
                             target_layers=[getattr(model, self.cam_target_layer)])
                     for model in self.models]

    def finish(self) -> None:
        # Grad-CAM installs forward and backward hooks on the model. Releasing them here rather
        # than leaving it to `BaseCAM.__del__` matters: that runs during interpreter shutdown,
        # where the module globals it needs may already be None, and it then raises a TypeError
        # into a finished run's output.
        for cam in self.cams:
            cam.activations_and_grads.release()
        self.cams = []

    # -- the two halves of the method ---------------------------------------

    def _union_cam(self, image: "Any", targets: List[int]) -> "Any":
        """Pixelwise maximum of one normalised, blurred Grad-CAM per model. [1, H, W]."""
        import torch
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from scipy.ndimage import gaussian_filter

        maps = []
        for cam, target in zip(self.cams, targets):
            heat = cam(input_tensor=image, targets=[ClassifierOutputTarget(target)])
            if self.cam_blur_sigma > 0:
                # Speckle removal only; the sigma is per-pixel, never across the batch axis.
                heat = gaussian_filter(heat, sigma=(0, self.cam_blur_sigma, self.cam_blur_sigma))
            low, high = heat.min(), heat.max()
            heat = (heat - low) / (high - low) if high > low else np.zeros_like(heat)
            maps.append(torch.from_numpy(heat).to(self.device))
        return torch.stack(maps, dim=0).max(dim=0)[0]

    def _soft_mask(self, union_cam: "Any") -> "Any":
        """Keep the top `cam_percentile`% of saliency, compress the tail. [1, 1, H, W]."""
        import torch

        cam = union_cam[0]
        threshold = torch.quantile(cam.flatten().float(),
                                   (100.0 - self.cam_percentile) / 100.0)
        if threshold.item() >= 1.0:
            return torch.zeros_like(cam)[None, None]
        soft = torch.clamp((cam - threshold) / (1.0 - threshold + 1e-8), 0, 1)
        return soft[None, None]

    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        import torch
        import torch.nn.functional as F
        from torch.amp import autocast

        source = Image.open(image_path).convert("RGB")
        # np.array, not np.asarray: PIL hands back a read-only buffer, and torch.from_numpy on a
        # read-only array is documented as undefined behaviour the moment anything writes to it.
        pixels = torch.from_numpy(np.array(source)).permute(2, 0, 1).float()[None] / 255.0
        pixels = pixels.to(self.device)
        mean, std = _stats_tensors(pixels.device, pixels.dtype)

        clean = (pixels - mean) / std
        amp = self.use_amp and self.device.startswith("cuda")

        # Each model's target is its own top-1 on the clean image, locked for the whole attack:
        # a moving target turns PGD into a random walk between classes.
        with torch.no_grad():
            targets = [int(model(clean)[0].argmax().item()) for model in self.models]

        union_cam = self._union_cam(clean, targets)
        soft_mask = self._soft_mask(union_cam)
        mask_coverage = float((soft_mask > 0).float().mean().item())

        adversarial = clean.clone().detach()
        loss_value = float("nan")
        for _ in range(self.num_steps):
            adversarial.requires_grad_(True)

            def ensemble_loss():
                losses = []
                for model, target in zip(self.models, targets):
                    logits = model(adversarial)
                    label = torch.full((logits.shape[0],), target,
                                       dtype=torch.long, device=logits.device)
                    losses.append(F.cross_entropy(logits, label))
                return torch.stack(losses).mean()

            # Forward in half precision where it is safe; the backward pass stays FP32, because
            # the sign of a gradient that underflowed to zero is zero, and the step is lost.
            if amp:
                with autocast(device_type="cuda"):
                    loss = ensemble_loss()
            else:
                loss = ensemble_loss()
            loss.backward()
            loss_value = float(loss.item())

            gradient = adversarial.grad.detach()
            step_direction = (gradient * soft_mask).sign()

            # Project in [0, 1] pixel space, not in normalised space: the epsilon budget is a
            # claim about pixels, and the per-channel std would otherwise stretch it unevenly.
            current = torch.clamp(adversarial.detach() * std + mean, 0, 1)
            stepped = current + self.alpha * step_direction
            delta = torch.clamp(stepped - pixels, -self.epsilon, self.epsilon)
            adversarial = ((torch.clamp(pixels + delta, 0, 1) - mean) / std).detach()

        final = torch.clamp(adversarial * std + mean, 0, 1)
        delta = (final - pixels).abs()

        array = (final[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        stats = {
            "target_class_indices": targets,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "num_steps": self.num_steps,
            "soft_mask_coverage": mask_coverage,
            "final_loss": loss_value,
            "max_abs_delta": float(delta.max().item()),
            "mean_abs_delta": float(delta.mean().item()),
        }
        return Image.fromarray(array), stats
