# Baselines

Five prior-work defenses, run on the same images and scored by the same attacker as VeilLens
itself. That is the whole design: the dataset, the attacker model and the output format all come
from the pipeline config you already use for `veillens-protect`, so the only thing that differs
between two rows of a comparison table is the defense.

```bash
veillens-baseline --method rand-noise --config configs/geolocation.yaml
veillens-score runs/geolocation/baselines/rand-noise
```

## The five

| Method | What it does | Key settings | Needs |
|---|---|---|---|
| `rand-noise` | Gaussian blur on a uniformly random 10% of pixels | `pixel_ratio 0.1`, `blur_ksize 51` | nothing |
| `ocr-blur` | EAST text detection, then Gaussian blur inside the boxes | `conf 0.2`, `max_ratio 0.4`, `blur_ksize 31`, 1920² input | EAST `.pb` (92 MB) |
| `mm-pgd` | Union-CAM-masked PGD against a Places365 ResNet-18 + ResNet-50 ensemble | **L∞ 8/255**, 20 steps, CAM percentile 20 | 2 checkpoints (136 MB), GPU |
| `geoshield` | Ensemble feature-matching PGD over three CLIP surrogates | **L∞ 8/255**, α 1/255, 100 steps, 1024² canvas | submodule + ~6 GB CLIP, GPU |
| `gpt-image` | One `gpt-image-2` edit per frame, from the round-1 reasoning prompt | `size auto`, `quality high` | `OPENAI_API_KEY`, paid |

Hyperparameters live in [`configs/baselines.yaml`](../configs/baselines.yaml), one section per
method, and nothing else. Each is short enough to read in full.

**The two adversarial methods are budget-matched at L∞ 8/255 deliberately.** They then differ by
7.7× in perceptual drift (LPIPS 0.035 vs 0.273), because `mm-pgd` concentrates that budget under a
saliency mask while `geoshield` spends it across the whole canvas. Matching the budget is what
makes that comparison possible, so do not change one epsilon without changing the other.

## Install

```bash
pip install -e '.[baselines]'                              # torch, torchvision, transformers,
                                                           # opencv, grad-cam, scipy
python scripts/fetch_baseline_weights.py --method all      # EAST + Places365, ~230 MB
git submodule update --init third_party/GeoShield          # if you did not clone --recursive
```

`fetch_baseline_weights.py` writes into `$VEILLENS_MODELS` (defaults to `$VEILLENS_DATA/models`)
and checks every file against a SHA-256 recorded from the copies the reported results were produced
with, so a truncated download or a substituted mirror fails there rather than three hours into a
run. `--check` verifies what is already on disk and downloads nothing.

`geoshield` additionally pulls three CLIP checkpoints from HuggingFace on first use — about 6 GB
into `$HF_HOME`, most of it LAION ViT-G/14. Nothing to do by hand; budget the disk.

Then confirm the install without spending anything:

```bash
veillens-baseline --method all --config configs/geolocation.yaml --dry-run
```

This resolves the manifest, every weight file and the submodule, and names whatever is missing
along with the command that fixes it. No GPU, no API call.

## Running

```bash
# one method
veillens-baseline --method mm-pgd --config configs/geolocation.yaml

# all five, four images each — a smoke test
veillens-baseline --method all --config configs/geolocation.yaml --limit 4

# compositional privacy instead
veillens-baseline --method rand-noise --config configs/compositional.yaml
```

Useful flags: `--limit N` (first N manifest entries), `--no-eval` (write protected images and
stop), `--no-resume` (re-protect images that already have output).

Output lands under the pipeline config's `output.base_dir`:

```
runs/geolocation/baselines/
├── _attacker_cache/                 attacker readings of the *unprotected* images, shared
│                                    across methods so five baselines pay for them once
└── rand-noise/
    ├── defended/<image_id>.png      the protected images
    ├── metadata/<image_id>.json     per-image statistics from the method itself
    ├── attacker_cache/              attacker readings of the protected images
    ├── evaluation_results/evaluation_results.json
    └── summary.json                 method, parameters, counts, per-image failures
```

`evaluation_results.json` is written in exactly the format the protection loop writes, so:

```bash
veillens-score runs/geolocation/baselines/rand-noise
```

works with no flags and no special case.

## How the measurement is done, and one way it differs from the loop

For every image the runner makes two attacker calls — one on the original, one on the protected
result — using the `evaluation:` model, the attack prompt and the attack schema from your pipeline
config. Both columns of the score table are therefore the same instrument reading two images.

**The protection loop does it slightly differently.** It has already read the original once, in
round 1, and reuses that reading for the unprotected column — a call made with the *defense* prompt
by the `reasoning:` model. So:

* the **protected** columns are attacker readings in both cases, and are directly comparable;
* the **unprotected** columns are two different measurements of the same images, and may differ.

When you compare a baseline against VeilLens, compare protected columns, or re-derive both
unprotected columns the same way. This is a property of the shipped loop, not something the runner
introduces, and it is written down here rather than smoothed over because a reader who diffs two
`evaluation_results.json` files will otherwise find it themselves and wonder.

`mask_pixels` and `mask_percentage` are filled from the actual pixel difference between the
original and the protected image, which makes the "how much of the frame did you touch" column
comparable across methods that have no mask at all.

## GeoShield: what is upstream's and what is not

GeoShield is the strongest geolocation-specific prior defense and the one most worth being precise
about. Upstream is <https://github.com/thinwayliu/Geoshield>, vendored as the
`third_party/GeoShield` submodule and pinned to commit `5001a82`.

**Upstream's, imported verbatim from the submodule:** the three CLIP feature extractors, the
ensemble feature loss, and its geo-semantic term. That is the attack objective — the part that
makes GeoShield GeoShield.

**Written here:** the iteration loop, in `GeoShield._attack_canvas`. Upstream's is wrapped in a
Hydra entry point, a Weights & Biases session, and a dataloader that scans a directory, none of
which can be pointed at a VeilLens manifest. The port is the same 100 sign-gradient steps on the
same objective; the correspondence is:

| upstream `geoshield.py` | here |
|---|---|
| `fgsm_attack_masked()` | `GeoShield._attack_canvas()` |
| `boxes = [full image]` when no bbox JSON is given | the only case; the random box choice and its area-weighted probabilities collapse to a constant, and `MaskCrop` becomes the identity |
| `Resize(res)` + `CenterCrop(res)` on the source | square tiling, below |
| `ImageFolderWithPaths` over two directories | the manifest, plus `pair_targets()` |
| `describe_image_placeholder()` | the `description:` config value |

Three consequences worth knowing before you read the numbers:

1. **It needs a target image per input.** This is a feature-*matching* attack. Upstream pairs by
   directory order; the pairing used for the published comparison did not survive.
   `pair_targets()` re-derives one deterministically — images are grouped by city (or by the
   leading token of the image id, which is the city for Cityscapes) and each group is paired with
   the next, so no image is ever paired with one from its own city. The chosen target is recorded
   in every image's `metadata/<image_id>.json`.
2. **Frames are split into square tiles.** Upstream's transform is `Resize(res)` then
   `CenterCrop(res)`, which is square: feed it a 2048×1024 frame and half the width is cropped
   away before the attack runs. Frames are therefore split along the longer axis into square tiles,
   attacked, and reassembled. At the default `input_res: 1024` a Cityscapes frame becomes two
   1024×1024 tiles and both transforms are the identity, so the perturbation lands on the true
   pixel grid and the result is pixel-aligned with the original — which is what makes SSIM against
   it mean anything. For a frame whose tiles are not already `input_res` square, the *perturbation*
   is resampled back to native size and re-clamped, never the image, so the L∞ bound stays exact
   and untouched pixels stay untouched. `metadata` records `tiles_resampled`.
3. **The text description is upstream's placeholder.** `geo_loss` wants a caption of the scene.
   Upstream ships `describe_image_placeholder()`, a stub with a TODO above it that returns
   `"A scenic outdoor photograph."` for every image, and the published comparison was run with that
   stub in place. It is exposed as `description:` in `configs/baselines.yaml` rather than hidden,
   but the shipped value is the stub's — changing it would no longer be the method that was
   measured.

## mm-pgd: the epsilon in the config

`configs/baselines.yaml` sets `epsilon: 0.03137` — 8/255, matching `geoshield`. The benchmark
config this was ported from carried `epsilon: 0.5`, which is half the dynamic range rather than an
adversarial budget; the tell is that its own comment still read `alpha: null # (auto:
epsilon/num_steps = 0.0015)`, which is 0.03/20, not 0.5/20. The budget had drifted and the
arithmetic was left behind. 8/255 is what the reported numbers were produced at.

The Places365 checkpoints load into stock `torchvision` ResNets with no missing or unexpected keys,
so no architecture is vendored here.

## What is not included

* **Perceptual metrics.** SSIM, LPIPS, CLIP-IQA and LIQE are reported in the paper but are not part
  of this package; the protected images are on disk in `defended/` and any standard implementation
  will reproduce them.
* **Other attackers.** The paper evaluates each defense against four attacker models. Here the
  attacker is whatever your config's `evaluation:` section names — swap the model there and rerun
  to get another column. The Gemini and Claude clients are not shipped.
* **`fuji-film` and `blue-noise`**, two further pixel-transform baselines that were run but are not
  in the paper's table.

For the `gpt-image` baseline both of the first two are archived rather than absent:
[`results/geolocation_500/`](../results/geolocation_500/) holds all four attackers' reasoning over
500 Cityscapes frames — 3,446 responses, raw and protected — plus per-image SSIM, LPIPS, CLIP-IQA
and LIQE. It is the only baseline that has been run end to end; that folder's README says exactly
which cells of the seven-condition matrix are still empty, and why.
