"""
`geoshield` — ensemble feature-matching PGD over three CLIP surrogates.

Upstream: <https://github.com/thinwayliu/Geoshield>, vendored as the `third_party/GeoShield`
submodule. **The attack objective is upstream's, imported verbatim** — the three CLIP feature
extractors, `EnsembleFeatureLoss`, and its `geo_loss` all come from `surrogates/`. What is written
here is the iteration loop (`_attack_canvas` below), because upstream's is wrapped in a Hydra
entry point, a Weights & Biases session, and a dataloader that scans a directory, none of which
can be pointed at a VeilLens manifest. `docs/BASELINES.md` maps the two line by line.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from veillens.baselines.base import Baseline

__all__ = ["GeoShield", "pair_targets"]

#: Upstream's `describe_image_placeholder()` return value. See the module docstring.
PLACEHOLDER_DESCRIPTION = "A scenic outdoor photograph."


def _group_of(item: Dict[str, Any]) -> str:
    """
    The 'somewhere else' key used to pick a target.

    A `city` field if the manifest carries one; otherwise the leading token of the image id,
    which is the city for Cityscapes ids like `aachen_000051_000019`. Falls back to the whole id,
    which just means every image is its own group and the rotation still terminates.
    """
    city = item.get("city")
    if city:
        return str(city)
    return str(item["image_id"]).split("_")[0]


def pair_targets(items: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Assign every item a target image path, from a different group where one exists.

    Deterministic given the manifest: items are sorted by id, grouped, and each group is paired
    with the next group in sorted order, cycling. With two or more groups no image is ever paired
    with one from its own, which matters because a same-city target would ask the attack to match
    features the image is already close to.
    """
    ordered = sorted(items, key=lambda entry: str(entry["image_id"]))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in ordered:
        groups.setdefault(_group_of(item), []).append(item)

    keys = sorted(groups)
    pairing: Dict[str, str] = {}
    for index, key in enumerate(keys):
        # With one group this is the group itself, so an image can be paired with itself; that
        # is a degenerate manifest rather than a case worth special-casing silently.
        donors = groups[keys[(index + 1) % len(keys)]]
        for position, item in enumerate(groups[key]):
            pairing[item["image_id"]] = donors[position % len(donors)]["image_path"]
    return pairing


def _tile_boxes(width: int, height: int, input_res: int) -> List[Tuple[int, int, int, int]]:
    """
    Split a frame along its longer axis into roughly square tiles.

    One tile per `short_side` of length, so a 2:1 frame gives two squares and a 1:1 frame gives
    one. Slices are equal width, which keeps reassembly exact — no overlap, no seam arithmetic.
    """
    long_side, short_side = max(width, height), min(width, height)
    count = max(1, round(long_side / short_side))
    edges = [round(index * long_side / count) for index in range(count + 1)]
    if width >= height:
        return [(edges[i], 0, edges[i + 1], height) for i in range(count)]
    return [(0, edges[i], width, edges[i + 1]) for i in range(count)]


class GeoShield(Baseline):
    name = "geoshield"
    summary = "100-step ensemble PGD over three CLIP surrogates (B/16, B/32, LAION ViT-G/14)"

    def __init__(self, backbones: List[str] | None = None, epsilon: float = 8.0,
                 alpha: float = 1.0, steps: int = 100, input_res: int = 1024,
                 crop_scale: Tuple[float, float] = (0.5, 0.9),
                 use_source_crop: bool = True, use_target_crop: bool = True,
                 description: str = PLACEHOLDER_DESCRIPTION, seed: int = 2023,
                 device: str = "cuda", **params: Any) -> None:
        super().__init__(backbones=backbones, epsilon=epsilon, alpha=alpha, steps=steps,
                         input_res=input_res, crop_scale=crop_scale,
                         use_source_crop=use_source_crop, use_target_crop=use_target_crop,
                         description=description, seed=seed, device=device, **params)
        self.backbones = backbones or ["B16", "B32", "Laion"]
        # epsilon and alpha are in 0-255 units, as upstream's config writes them: its tensors
        # are un-normalised 0-255 floats, so `epsilon: 8` is the 8/255 bound the paper reports.
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.input_res = input_res
        self.crop_scale = tuple(crop_scale)
        self.use_source_crop = use_source_crop
        self.use_target_crop = use_target_crop
        self.description = description
        self.seed = seed
        self.device = device
        self.extractor = None
        self.loss = None
        self._targets: Dict[str, str] = {}

    @classmethod
    def check_prerequisites(cls, params: Dict[str, Any]) -> List[str]:
        from veillens import paths

        root = paths.geoshield_root()
        if not (root / "surrogates").is_dir():
            return [
                f"missing GeoShield checkout: {root}\n"
                "      git submodule update --init third_party/GeoShield\n"
                "      (or point VEILLENS_GEOSHIELD at an existing clone)"
            ]
        return []

    def prepare(self, items: List[Dict[str, Any]]) -> None:
        from veillens.paths import bootstrap_geoshield

        bootstrap_geoshield()
        # Imported from the submodule, not reimplemented: these are the attack objective.
        from surrogates import (  # type: ignore[import-not-found]
            ClipB16FeatureExtractor,
            ClipB32FeatureExtractor,
            ClipL336FeatureExtractor,
            ClipLaionFeatureExtractor,
            EnsembleFeatureExtractor,
            EnsembleFeatureLoss,
        )

        available = {
            "B16": ClipB16FeatureExtractor,
            "B32": ClipB32FeatureExtractor,
            "L336": ClipL336FeatureExtractor,
            "Laion": ClipLaionFeatureExtractor,
        }
        unknown = [name for name in self.backbones if name not in available]
        if unknown:
            raise ValueError(f"unknown GeoShield backbone(s) {unknown}; "
                             f"available: {sorted(available)}")

        models = [available[name]().eval().to(self.device).requires_grad_(False)
                  for name in self.backbones]
        self.extractor = EnsembleFeatureExtractor(models)
        self.loss = EnsembleFeatureLoss(models)
        self._targets = pair_targets(items)

    # -- the attack ---------------------------------------------------------

    def _attack_canvas(self, source: "Any", target: "Any") -> "Any":
        """
        Upstream's `fgsm_attack_masked`, for the single-full-image-box case.

        Both tensors are [1, 3, res, res] in 0-255. Returns the perturbation, same shape, already
        clamped to the epsilon ball. Without a bounding-box JSON upstream's box list is just the
        full frame, so its random box choice and area-weighted probabilities collapse to a
        constant and the mask crop is the identity -- that is the only simplification here.
        """
        import torch
        from torchvision import transforms

        delta = torch.zeros_like(source, requires_grad=True)
        source_crop = (transforms.RandomResizedCrop(self.input_res, scale=self.crop_scale)
                       if self.use_source_crop else torch.nn.Identity())
        target_crop = (transforms.RandomResizedCrop(self.input_res, scale=self.crop_scale)
                       if self.use_target_crop else torch.nn.Identity())
        local_crop = transforms.RandomCrop(224)

        for _ in range(self.steps):
            # Re-set every step, not hoisted: the geo-semantic block below overwrites
            # ground_truth with a fresh random crop of the *source*, so this restores the
            # target as the match objective at the top of each iteration.
            with torch.no_grad():
                self.loss.set_ground_truth(target)

            adversarial = source + delta
            global_features = self.extractor(source_crop(adversarial))
            local_features = self.extractor(local_crop(adversarial))

            # Negative similarity: move the embedding toward the target's.
            objective = -(self.loss(global_features) + self.loss(local_features))

            with torch.no_grad():
                self.loss.set_ground_truth(target_crop(source))
                self.loss.set_geotext_truth(self.description, self.device)
            objective = objective - (self.loss.geo_loss(global_features)
                                     + self.loss.geo_loss(local_features))

            gradient = torch.autograd.grad(objective, delta, create_graph=False)[0]
            delta.data = torch.clamp(delta + self.alpha * gradient.sign(),
                                     min=-self.epsilon, max=self.epsilon)

        return delta.detach()

    def apply(self, image_path: str, image_id: str) -> Tuple[Image.Image, Dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        target_path = self._targets[image_id]
        source_image = Image.open(image_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Seeded per image: the attack draws a random crop at every step, so without this the
        # same command on the same inputs gives a different perturbation each time.
        torch.manual_seed(self.seed)

        # np.array, not np.asarray: PIL's buffer is read-only, and torch.from_numpy over one is
        # undefined behaviour as soon as anything writes through the tensor.
        source = torch.from_numpy(np.array(source_image)).permute(2, 0, 1).float()[None]
        source = source.to(self.device)
        target = torch.from_numpy(np.array(target_image)).permute(2, 0, 1).float()[None]
        target = target.to(self.device)

        width, height = source_image.size
        boxes = _tile_boxes(width, height, self.input_res)
        target_boxes = _tile_boxes(*target_image.size, self.input_res)

        protected = source.clone()
        resampled = False
        for index, (left, top, right, bottom) in enumerate(boxes):
            tile = source[:, :, top:bottom, left:right]
            t_left, t_top, t_right, t_bottom = target_boxes[index % len(target_boxes)]
            target_tile = target[:, :, t_top:t_bottom, t_left:t_right]

            native = tile.shape[-2:]
            canvas = tile
            if native != (self.input_res, self.input_res):
                resampled = True
                canvas = F.interpolate(tile, size=(self.input_res, self.input_res),
                                       mode="bicubic", align_corners=False)
            target_canvas = F.interpolate(target_tile, size=(self.input_res, self.input_res),
                                          mode="bicubic", align_corners=False)

            delta = self._attack_canvas(canvas, target_canvas)

            # The perturbation, not the perturbed canvas, is what comes back to native
            # resolution: resampling the image would blur every pixel the attack never touched,
            # and resampling the delta keeps the L-infinity bound exact after the re-clamp.
            if native != (self.input_res, self.input_res):
                delta = F.interpolate(delta, size=native, mode="bicubic", align_corners=False)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            protected[:, :, top:bottom, left:right] = torch.clamp(tile + delta, 0.0, 255.0)

        difference = (protected - source).abs() / 255.0
        array = protected[0].permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
        stats = {
            "target_image": target_path,
            "backbones": list(self.backbones),
            "epsilon_255": self.epsilon,
            "alpha_255": self.alpha,
            "steps": self.steps,
            "input_res": self.input_res,
            "tiles": len(boxes),
            "tiles_resampled": resampled,
            "description": self.description,
            "max_abs_delta": float(difference.max().item()),
            "mean_abs_delta": float(difference.mean().item()),
        }
        return Image.fromarray(array), stats
