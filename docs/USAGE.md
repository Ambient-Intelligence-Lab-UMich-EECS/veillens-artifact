# Usage

Four commands ship: `veillens-protect` runs the loop, `veillens-baseline` runs a prior-work
defense on the same images, `veillens-score` reads what either of them wrote, and
`veillens-verify` turns a finished run into a per-image verdict.

## `veillens-score` — no key, no GPU, no data

Reads the JSON a run already produced and prints the aggregate table. This is the fastest way to
see what the pipeline emits, and it works straight from a clone — `examples/results/` ships two
images per setting, taken from real runs:

```bash
veillens-score examples/results/geolocation
veillens-score examples/results/compositional
veillens-score examples/results/social_relation
```

The setting is sniffed from the error key the results were written under, so `--setting` is only
needed to override it.

For geolocation it reports the fraction of images the attacker still localises within each
distance threshold, before and after protection. For compositional privacy it reports the mean
risk score before and after, and the relative reduction. For social relationship it reports
pairwise accuracy, six-way and three-way, over two denominators.

## `veillens-verify` — no key, no GPU, no data

Where `veillens-score` reports a distribution, this reports a verdict: whether the cue
granularities the run promised to remove are actually gone.

```bash
veillens-verify runs/geolocation/my_run
veillens-verify examples/results/geolocation
```

The bundled examples carry their own `eval_cache/`, so both criteria are computed with no extra
flag. `--responses` exists for a run whose attacker responses sit elsewhere, or one evaluated by
several attackers — the residual criterion is then computed against whichever you name:

```bash
veillens-verify runs/geolocation/my_run \
    --responses runs/geolocation/my_run/evaluation/gpt-5.2/responses
```

The archived trees under `results/` ship the raw responses and a per-image `index.csv`, but not
the `evaluation_results.json` those numbers were derived from, so they are read as tables rather
than verified.

Two criteria per image, both derived from `experiment.mask_level` and reported separately: the
attacker must be at least the target rung's distance from the truth (10 km under city-level
protection), **and** its own report must name no surviving cue at a protected granularity. A
criterion that could not be computed is reported as such and never counts as a pass.

A live run writes the same report itself, to `evaluation_results/verification.json`. The full
model — the granularity ladder, why there are two criteria, and what the attacker prompt has to
allow for the second one to mean anything — is in [VERIFICATION.md](VERIFICATION.md).

## `veillens-protect --dry-run` — validate an install

```bash
veillens-protect --config configs/geolocation.yaml --dry-run
```

Resolves and checks everything the run would need — config sections, prompt and schema files,
model checkpoints, the Grounded-SAM-2 submodule and its compiled GroundingDINO extension, the
dataset manifest — then exits without making an API call or touching a GPU. It exits non-zero and
lists what is missing, with the command to fix each one.

The extension check is there because its absence is not a loud failure. A checkout without the
built `_C` object imports cleanly and loads weights cleanly, then raises on the first detection.
The loop catches that per clue, records "no ground", and runs to completion — reporting success
and a full score table computed by re-reading images nothing ever edited. The loop now refuses to
finish that way (it raises if clues were selected and none were grounded), but the dry run is what
catches it before you have spent an hour and a few hundred API calls.

## `veillens-protect` — run the loop

```bash
veillens-protect --config configs/geolocation.yaml
```

Needs `OPENAI_API_KEY`, a CUDA GPU, the model weights, the images, and the submodule's compiled
extensions (see the install section of `README.md`). See `docs/DATA.md` for the data.

## The config

One file per privacy setting, in `configs/`. Every key is listed here.

### `dataset`

| Key | Meaning |
|---|---|
| `dataset_json_file` | Path to the manifest. Schema v2: a mapping with `schema_version: 2` and an `items` list, where each item has `image_id`, `dataset`, `relpath`, and the setting's ground-truth field. `dataset`/`relpath` are joined against `$VEILLENS_DATA`, so a manifest carries no absolute paths |

### `experiment`

| Key | Meaning |
|---|---|
| `privacy_class` | `geolocation`, `compositional` or `social_relation`. Determines the schema, the prompts, the evaluation metric, and what `mask_level` means. `personal_info` and `pi` are aliases for `compositional`; `relationship` and `social_relationship` for `social_relation` |
| `mask_level` | **geolocation**: any subset of `["country level clue", "city level clue", "street level clue"]`. **compositional**: an integer 1–4; level N removes cues of severity ≤ N (1 = severe). **social_relation**: any subset of `["intimate level cue", "non-intimate level cue"]` |
| `max_iter` | Round-2 iterations after the round-1-only pass. The run writes up to `max_iter + 1` protected images: iteration 0 removes round 1's cues, then one image per verification gate |
| `gate_requires_residual_clear` | Also require the oracle to report no surviving cue at a protected granularity before an image may stop. Default `false`; strictly stronger than the distance gate, and strictly more iterations |

There is no separate stop-distance key. The threshold is derived from `mask_level`: the coarsest
protected granularity picks a rung on the ladder, and the rung fixes the distance — street → 1 km,
city → 10 km, country → 100 km. `experiment.val_dist` was that key and is now rejected at load, so a
stale config fails loudly instead of running under a threshold that no longer matches what it
removes. See [VERIFICATION.md](VERIFICATION.md).

### `prompts`

Six paths, repo-relative: `round1_{schema,prompt}`, `round2_{schema,prompt}`, `eval_{schema,prompt}`.
Round 1 discovers cues, round 2 re-reads the edited image, and the eval pair is the attacker that
produces the reported numbers. Omit the section to take the defaults from the privacy-class spec.

### `round2_regions`

| Key | Meaning |
|---|---|
| `fine_grained` | `false` verifies the whole image each iteration. `true` runs the multi-scale residual search over sliding-window crops instead |
| `crop_width`, `crop_height` | Crop size for the sliding-window search |
| `overlap_percent` | Overlap between adjacent crops (`0.1` = 10%) |
| `upscale_target` | Longer dimension each crop is scaled to before it is sent to the oracle |

### `models`

Checkpoint paths and HuggingFace model IDs. `grounded_sam2_root` defaults to
`third_party/Grounded-SAM-2`; `VEILLENS_GSAM2` redirects it to an existing checkout.

### `detection`

`box_threshold` and `text_threshold` are Grounding DINO's. `top_k_boxes` caps how many detection
boxes are kept per cue after target-concept filtering. A cue whose object is not detected above
threshold is left unmasked rather than force-matched.

`use_grounding_dino` (default `true`) turns open-set detection off. A setting whose regions come
from its dataset — `social_relation` reads PISC's person boxes — never calls the detector, so with
this `false` its weights are not loaded and its compiled CUDA extension is not required. `--dry-run`
stops checking for both.

### `intervention`

Only read by settings that ground from dataset boxes; absent from the other configs.

| Key | Meaning |
|---|---|
| `mask_scope` | `face` cuts the person mask SAM 2 returned down to its top band; `person` inpaints the whole silhouette |
| `head_width_multiple` | Band height as a multiple of the head's width, read off the mask's topmost rows |
| `head_fraction` | Floor on that height, as a share of the mask's own height |
| `width_fraction` | Share of the head rows' width, centred. `1.0` (the default) clips nothing — the band is already confined to the person by intersection |

SAM 2 is always prompted with the dataset's whole person box; the narrowing happens on the mask.
That is not interchangeable with narrowing the box: an annotation box's top band is only the head
for a subject standing upright, and on someone crouched or bent forward it is background and
shoulder.

The band's height is the larger of the two estimates above. A fixed share of the figure's height
cannot serve both poses — a head is a small fraction of a standing silhouette and a large one of a
crouched silhouette — so the primary rule scales with the head's own width, which does not change
with pose, and the fraction is kept only as a floor. The mask is also clipped to the subject's own
annotation box before the band is cut, so a mask that bleeds past the person cannot mask their
neighbour.

**Known limitation.** This is geometry over a segmentation, not face detection, and it is only as
good as the silhouette SAM 2 returns. On clean, upright, well-separated subjects it lands on the
face; where SAM 2's person mask bleeds into a patterned garment or the sky behind it, the band
inherits that and the edit is both larger than a face and off-centre. Spot-check
`intermediate/<image>/round1/` before trusting a large run, and treat a face detector inside the
dataset box as the upgrade path if the heuristic proves too coarse for your images.

### `generation`

SDXL parameters: `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale`,
`strength`, `generator_seed`. `positive_prompt_prefix` is prepended to every inpainting prompt and
therefore sets the visual style of edited regions.

### `performance`

`use_fp16`, `use_tf32`, `enable_attention_slicing`, `enable_vae_tiling`. VAE tiling matters most:
it costs a few percent in speed and saves roughly 40% of peak memory on large images.

### `reasoning` and `evaluation`

Two sections, deliberately separate. `reasoning` is the **defense oracle** — it discovers cues,
analyses residuals, and decides whether to iterate. `evaluation` is the **attacker** whose numbers
are reported. Keeping them apart is what makes the measurement independent of the defense.

### `output`

`base_dir` is the run directory.

## Building a manifest

Geolocation and social relationship build theirs from a dataset checkout; the compositional
manifest ships in `manifests/`.

```bash
python -m veillens.datasets.cityscapes \
    --cityscapes-root $VEILLENS_DATA/cityscapes \
    --output $VEILLENS_DATA/cityscapes/dataset.json --images-per-city 25

python -m veillens.datasets.pisc \
    --pisc-root $VEILLENS_DATA/pisc \
    --annotated-dir $VEILLENS_DATA/pisc_annotated \
    --num-images 500 --seed 0 \
    --output manifests/social_relation_500.json
```

The PISC builder also writes the box overlay each image is evaluated with — a coloured rectangle
labelled `P1`, `P2`, … per annotated subject — because a prompt that asks about "P1 and P2" is
meaningless without it. It reports what it skipped and records the same counts in the manifest's
`provenance` block; see [DATA.md](DATA.md) for why that matters.

## The output tree

```
<base_dir>/
├── original_cache/               round-1 oracle responses, keyed by image_id
├── eval_cache/                   attacker responses
├── evaluation_results/           the aggregate veillens-score reads
├── phase_cache/                  resume state
├── depth/<image>.png             MiDaS depth map, computed once per image
├── resized_inputs/<image>.jpg    only when the setting caps input resolution — see below
└── intermediate/<image>/
    ├── original.png              the input as processed (downscaled, if the setting caps it)
    ├── round1/                   the grey composite: what round 1 decided to remove. Diagnostic
    ├── segmentation/             per-cue masks and overlays
    ├── working/                  the image as it is being edited
    └── final/                    iter0…iterN snapshots, and <image>_final.png — the inpainted
                                  result, and the image the attacker is scored on
```

A run is resumable: `phase_cache/` records which images finished, and re-running the same config
skips them.

### Input resolution

Nothing in the edit path bounds resolution: SDXL inpaints at whatever size it is handed, so a
multi-megapixel source asks for a single allocation large enough to exhaust a 48 GB GPU. A privacy
setting whose dataset ships full-frame camera originals therefore declares a cap, as
`max_input_dim` on its `PrivacyClassSpec`.

**Compositional caps the long side at 1024**, because VISPR ships unresized originals up to ~10
megapixels. Geolocation (Cityscapes, 2048×1024) and social relationship (PISC, web-sized) declare
no cap; their images are already bounded.

The scaling happens once, before anything reads the image: the downscaled copy is written to
`resized_inputs/` and becomes the run's input, so the masks, the depth map and the attacker's
reading are all in one coordinate space. An image already within the cap is used verbatim and is
not re-encoded.

## `veillens-baseline` — the same images, the same attacker, a different defense

```bash
veillens-baseline --method all --config configs/geolocation.yaml --dry-run
veillens-baseline --method rand-noise --config configs/geolocation.yaml --limit 4
veillens-score runs/geolocation/baselines/rand-noise
```

Five methods: `rand-noise`, `ocr-blur`, `mm-pgd`, `geoshield`, `gpt-image`, or `all`. The pipeline
config supplies the dataset, the attacker and the output root; the method's own hyperparameters
come from `configs/baselines.yaml`. Output lands in `<base_dir>/baselines/<method>/` and carries an
`evaluation_results/evaluation_results.json` in the same format as above, so `veillens-score` reads
it unchanged.

Full details, including which parts of the GeoShield baseline are upstream's and which are not:
[BASELINES.md](BASELINES.md).
