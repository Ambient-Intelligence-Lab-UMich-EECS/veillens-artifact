# Usage

Two commands ship: `veillens-protect` runs the loop, `veillens-score` reads what a run wrote.

## `veillens-score` — no key, no GPU, no data

Reads the JSON a run already produced and prints the aggregate table. This is the fastest way to
see what the pipeline emits, and it works straight from a clone:

```bash
veillens-score examples/results/geolocation/evaluation_results.json
veillens-score examples/results/compositional/run_evaluation_results.json --setting compositional
```

For geolocation it reports the fraction of images the attacker still localises within each
distance threshold, before and after protection. For compositional privacy it reports the mean
risk score before and after, and the relative reduction.

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
| `privacy_class` | `geolocation` or `compositional`. Determines the schema, the prompts, the evaluation metric, and what `mask_level` means. `personal_info` and `pi` are accepted aliases for `compositional` |
| `mask_level` | **geolocation**: any subset of `["country level clue", "city level clue", "street level clue"]`. **compositional**: an integer 1–4; level N removes cues of severity ≤ N (1 = severe) |
| `max_iter` | Round-2 iterations after the round-1-only pass. The run writes up to `max_iter + 1` protected images: iteration 0 removes round 1's cues, then one image per verification gate |
| `val_dist` | **geolocation only.** Stop iterating an image once the oracle's prediction is at least this far off, in km. The compositional setting has no distance-based stop and always runs the full budget |

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

## The output tree

```
<base_dir>/
├── original_cache/               round-1 oracle responses, keyed by image_id
├── eval_cache/                   attacker responses
├── evaluation_results/           the aggregate veillens-score reads
├── phase_cache/                  resume state
├── depth/<image>.png             MiDaS depth map, computed once per image
└── intermediate/<image>/
    ├── original.png              the input, copied
    ├── round1/                   after round 1's edits
    ├── segmentation/             per-cue masks and overlays
    ├── working/                  the image as it is being edited
    └── final/                    iter0…iterN snapshots, and <image>_final.png
```

A run is resumable: `phase_cache/` records which images finished, and re-running the same config
skips them.
