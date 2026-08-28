# Geolocation, 500 images, defence baselines — VLM reasoning

Every VLM response behind the 500-image Cityscapes geolocation experiment, one file per response.
**Reasoning only — no images are included.**

> **Produced before the granularity verification change.** This run used the superseded
> `experiment.val_dist: 5.0` stop rule and the earlier attacker prompt, which required the attacker
> to report at least one street-level clue. The current code derives its stop threshold from
> `experiment.mask_level` instead — city-level protection stops at **10 km**, not 5 — and the
> attacker prompt now permits an empty clue list. Both the defense's behaviour and the measuring
> instrument therefore differ from what produced the numbers below, and these responses cannot be
> used to evaluate the residual-cue criterion. See [docs/VERIFICATION.md](../../docs/VERIFICATION.md).

The experiment is a matrix: seven image conditions — the unprotected frame, VeilLens, and the five
baselines of [`docs/BASELINES.md`](../../docs/BASELINES.md) — each read by four independent VLMs.
**Eight of its twenty-eight cells exist.** What has been run end to end is the unprotected reference
and the **`gpt-image`** baseline, both scored by all four evaluators. The other twenty cells are
directories holding a `PENDING.json` that says what belongs there and why it is absent; they are
placeholders, not findings.

```
results/geolocation_500/
├── MANIFEST.json                       provenance, counts, and the checks below
├── README.md
├── index.csv                           500 rows: ids, ground truth, every error, image quality
├── perceptual_metrics.json             SSIM / LPIPS / CLIP-IQA / LIQE for the 500 edited pairs
├── prompts/                            what the four evaluators were given, verbatim
│   ├── attack_prompt.txt
│   └── eval_schema.json
├── process/                            part 1 — reasoning during protection
│   └── PENDING.json                      not run on this image set (see below)
└── evaluation/                         part 2 — reading the finished image
    ├── raw/                              the unprotected frame
    │   ├── gpt-5.2/                        summary.json + responses/<image_id>.json   500
    │   ├── gemini-3-flash-preview/                                                    500
    │   ├── gemini-3.1-pro-preview/                                                    100
    │   └── gemini-2.5-flash/                                                          500
    ├── gpt-image/                        one gpt-image-2 edit per frame
    │   ├── gpt-5.2/                                                                   500
    │   ├── gemini-3-flash-preview/                                                    498
    │   ├── gemini-3.1-pro-preview/                                                    348
    │   └── gemini-2.5-flash/                                                          500
    ├── veillens/PENDING.json             the method itself — never run on these 500
    ├── rand-noise/PENDING.json
    ├── ocr-blur/PENDING.json
    ├── mm-pgd/PENDING.json
    └── geoshield/PENDING.json
```

500 images, **3,446 responses**. 99 images have all eight, 248 have seven, 153 have six.

Condition directories are named for `veillens-baseline --method`, so a finished run drops into the
slot already waiting for it.

## The image set

500 Cityscapes frames, **25 from each of 20 cities**, pinned by
`readydata/dataset_random_500/metadata/dataset.json` and reproduced in `index.csv`.

It is the 100-image set of [`results/geolocation_100/`](../geolocation_100/) plus 400 additions.
The two halves are disjoint and their union is exactly these 500 — checked. `index.csv` marks the
overlap in `in_geolocation_100`.

That split is not cosmetic. The two halves were scored in **separate runs against separate
manifests**, because the 100-image half's calls were already paid for and were not repeated. Every
number in this folder is assembled from both.

## Part 1 — reasoning during the process: not run

`process/` is empty. **No VeilLens protection run covers this image set.** The largest Cityscapes
geolocation run on disk is 100 images (`pipeline2/result/test_iter3`), and those 100 ids are exactly
the 100-image half of this set; its reasoning is archived in full at
[`results/geolocation_100/`](../geolocation_100/). Nothing on the filesystem extends past it.

The consequence is worth stating before any table is read: **nothing in this folder measures
VeilLens.** Everything here is one baseline against the unprotected reference. The `veillens/`
evaluation cell is empty for the same reason — there is no protected image for these 500 for anyone
to score.

## Part 2 — evaluation

`evaluation/<condition>/<evaluator>/responses/<image_id>.json` is one model's read of one image
under one condition: its country → city → street reasoning and a latitude-longitude guess with a
confidence. `summary.json` beside it carries that cell's aggregate and its provenance.

### What is filled, and what is not

| Condition | What it is | Status |
|---|---|---|
| `raw` | the unmodified Cityscapes frame | **run** — 4 evaluators, 1,600 responses |
| `gpt-image` | one `gpt-image-2` edit per frame, from a fixed prompt | **run** — 4 evaluators, 1,846 responses |
| `veillens` | the method itself | not run on this image set — no protected images exist |
| `rand-noise` | Gaussian blur on 10% of pixels | not run — implementation exists, no defended images on disk |
| `ocr-blur` | EAST text detection, then blur inside the boxes | not run — `dataset_random_100/ocr-blur/` exists and is empty |
| `mm-pgd` | CAM-masked PGD against a Places365 ensemble | not run — `dataset_random_100/mm-pgd/` exists and is empty |
| `geoshield` | ensemble feature-matching PGD over three CLIP surrogates | **started, not finished** — 200 half-image tiles for the 100-image subset exist under `third_party/Geoshield/output/`; nothing stitched, nothing evaluated |

## The baseline under test

`gpt-image-2`, `quality: high`, `size: auto`, one call per image from
[`prompts/geolocation/defense_image_edit_prompt.txt`](../../prompts/geolocation/defense_image_edit_prompt.txt)
— the same file that ships in this repository. No reasoning stage, no grounding, no iteration: it is
the closest thing here to *just ask a good image model to remove the location*. All 500 images
returned an edit; **zero refusals, zero failures**.

**The edit changes the resolution, and that is a confound.** Every input is 2048×1024; every output
came back 1774×887 (one at 1773×887). So each evaluator read a materially smaller picture in the
`gpt-image` condition than in `raw`, and the perceptual metrics compare a resized pair. Some part of
the protection measured below is resolution loss rather than content removal, and **nothing in this
run separates the two.** A control — the raw frame downscaled to 1774×887 and scored — does not
exist.

## The evaluators

Four models, each with its id recorded in the source run's `metadata.model` — unlike
[`results/geolocation_100/`](../geolocation_100/), attribution here rests on the artifacts rather
than on a config default. **None of them is inside a protection loop**, because no loop was run on
this set, so every number below is a transfer number.

They were given `prompts/attack_prompt.txt` and `prompts/eval_schema.json`, copied here verbatim.
That is the **benchmark's** attacker prompt, and it is not the same file as
[`prompts/geolocation/attack_base_prompt.txt`](../../prompts/geolocation/attack_base_prompt.txt) at
the top of this repository, which is the pipeline's. The benchmark prompt asks only for reasoning
and a location estimate, which is why these responses carry two keys and no `critical_features`.

## The numbers these files produce

Percentage of images located within each threshold (strictly `<`). **Higher is worse protection.**

| | 0.5 km | 1 km | 5 km | 10 km | 25 km | 100 km | n | mean | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **raw**, GPT-5.2 | 16.8 | 29.2 | 45.6 | 46.2 | 50.0 | 65.6 | 500 | 108.7 | 25.14 km |
| **raw**, Gemini 3 Flash | 40.6 | 55.4 | 67.6 | 68.6 | 72.6 | 85.0 | 500 | 46.4 | 0.78 km |
| **raw**, Gemini 3.1 Pro | 49.0 | 57.0 | 66.0 | 67.0 | 72.0 | 82.0 | *100* | 43.5 | 0.61 km |
| **raw**, Gemini 2.5 Flash | 15.0 | 29.0 | 45.8 | 47.2 | 53.4 | 72.4 | 500 | 86.3 | 18.19 km |
| `gpt-image`, GPT-5.2 | 4.8 | 8.6 | 16.6 | 17.0 | 19.6 | 40.2 | 500 | 197.5 | 162.83 km |
| `gpt-image`, Gemini 3 Flash | 14.3 | 22.3 | 32.7 | 33.7 | 37.3 | 55.0 | 498 | 131.2 | 70.42 km |
| `gpt-image`, Gemini 3.1 Pro | 13.8 | 18.7 | 27.3 | 27.9 | 30.7 | 42.2 | *348* | 164.3 | 139.42 km |
| `gpt-image`, Gemini 2.5 Flash | 7.6 | 15.8 | 26.6 | 27.6 | 31.2 | 54.4 | 500 | 153.9 | 72.03 km |

The two italicised `n`s are not 500 and those rows are **not on the same sample as the others** —
see the quota gap below.

The raw rows are the finding to read first: **Gemini 3 Flash puts the median image within 780 m of
its true position**, and Gemini 3.1 Pro within 610 m, on ordinary dashcam frames with no landmark
requirement. GPT-5.2 and Gemini 2.5 Flash are far weaker attackers on this set. Any claim about a
defence is a claim relative to whichever of these attackers is being run.

### Paired, on the images each evaluator scored in both conditions

| Evaluator | n | raw median | edited median | < 25 km, raw → edited | further | closer | unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | 500 | 25.14 km | 162.83 km | 50.0% → 19.6% | 331 | 130 | 39 |
| Gemini 3 Flash | 498 | 0.78 km | 70.42 km | 72.5% → 37.3% | 377 | 121 | 0 |
| Gemini 3.1 Pro | 100 | 0.61 km | 129.65 km | 72.0% → 29.0% | 76 | 19 | 5 |
| Gemini 2.5 Flash | 500 | 18.19 km | 72.03 km | 53.4% → 31.2% | 316 | 183 | 1 |

The edit helps against every evaluator, and it is nowhere near a fix. Gemini 3 Flash still places
**a third of the edited images inside 25 km**, and on 121 of 498 images it located the edited photo
*better* than the original. "Unchanged" is the model returning the same rounded coordinate for both
pictures — a city-centroid guess the edit never disturbed; GPT-5.2 does this on 39 images.

Median is the honest column here. The means are dragged by a long tail of images placed in the wrong
country, and a defence that moves one guess from Aachen to Portugal buys a large mean improvement
for no privacy that a 25 km threshold would not already have granted.

## The quota gap

`gemini-3.1-pro-preview` is incomplete, and not for a reason that re-reading the archives can fix:
the model has a **250-requests-per-model-per-day cap**. Its 800 calls for the 400-image half were
scheduled as one job; it exhausted the day's quota part way through the `gpt-image` set and **never
reached the `raw` set at all**. Every unscored call returned HTTP 429 with a null error and a null
location estimate, so there is no reasoning to archive and no file is emitted.

| | scored | unscored | covers |
|---|---:|---:|---|
| `gpt-image` / Gemini 3.1 Pro | 348 | 152 | the 100-image half, plus 248 of the 400 |
| `raw` / Gemini 3.1 Pro | 100 | 400 | the 100-image half only |

Its raw-versus-protected comparison is therefore over **the 100 images both cover**, not over 500,
and its two rows in the table above are on different samples from each other. The other gap is
smaller and ordinary: two API failures for Gemini 3 Flash on the edited images
(`aachen_000083_000019`, `weimar_000000_000019`), both named in the runs' own `failures` lists.

Blank cells in `index.csv` always mean the call failed. No response file was dropped — checked.

## Image quality

`perceptual_metrics.json` holds one row per edited pair, all 500, plus the aggregate.

| | mean | sd | median |
|---|---:|---:|---:|
| SSIM ↑ | 0.859 | 0.069 | 0.880 |
| LPIPS ↓ | 0.139 | 0.044 | 0.132 |
| CLIP-IQA, edited − original | +0.158 | 0.058 | +0.150 |
| LIQE, edited − original | +0.677 | 0.432 | +0.613 |

Both no-reference quality scores go **up** after the edit: the generative model returns a cleaner,
more photographic picture than the Cityscapes dashcam frame it was given. Read alongside the
resolution change, that is a warning about using no-reference IQA as a fidelity check at all — it
rewards exactly the kind of resynthesis that destroys the evidence being measured.

**SSIM for the first 100 images is a recompute, not the archive.** That half ran under
Python 3.14 + numpy 2.2.6, which silently corrupts skimage's SSIM above 256 KiB — its archived
`ssim_mean` reads 37.5, impossible for a bounded metric. Those 100 values come from
`ssim_recomputed_numpy2.3.5.json` instead; the archived number is kept beside each row as
`ssim_archived`, and `ssim_source` says which is which.

## Response shape

All four evaluators return exactly the two keys of `prompts/eval_schema.json`:

```json
{
  "reasoning": "country / city / street reasoning, in prose",
  "location_estimate": {"latitude": 51.452, "longitude": 7.012,
                        "confidence": "Medium", "region": "Germany, North Rhine–Westphalia, Essen"}
}
```

No `critical_features` and no `image_path` — the benchmark's attacker prompt does not ask for
either. Which image a response read is fixed by its filename, and that came from the alignment
below.

## Which image each response belongs to

The source stores its answers as a **positional array with no image id**. Slot *i* belongs to image
*i* of the manifest named in that run's archived `experiment_config.yaml` — `dataset.json` for the
100-image half, `dataset_new400.json` for the 400-image half. Every list length matched its
manifest exactly, so nothing was guessed; a mismatch would have stopped the build.

One run, `gpt_image_100_20260813_122752`, predates configs being archived. Its lists are 100 long,
which matches `dataset.json` exactly, so the fallback is unambiguous.

The mapping is confirmed by evidence that was not used to build it: recomputing
`haversine(ground truth, location_estimate)` for every one of the 3,446 responses reproduces the
recorded `prediction_errors_km` **exactly** — zero difference, not merely small — at R = 6371.0 km.
A misalignment would have to survive 3,446 exact coincidences.

Answers are unioned by `image_id` across every matching run, any non-null answer winning, which is
the same rule as `benchmark/analysis/merge_gpt_image_500.py`. A 10-image smoke run
(`gpt_image_10_20260813_103949`) is excluded: it is a subset of the 100 that the 100-image run
already covers.

## What was checked

All 14 passed; `MANIFEST.json` records them in machine-readable form.

- the two halves partition the 500-image set — 100 + 400, disjoint, union equals `dataset.json`
- every emitted response parses as JSON with a non-empty `reasoning` — 3,446/3,446
- every response carries exactly the keys of `prompts/eval_schema.json` — 3,446/3,446
- emitted bodies are deep-equal to the source objects — 3,446, zero differences
- recomputed `haversine(gt, prediction)` equals the recorded error — 3,446/3,446, **exactly**
- `index.csv` lists the 500 once, in `dataset.json` order
- every `index.csv` error cell equals the source's `prediction_errors_km` exactly — 3,446 cells,
  written at full round-trip precision rather than rounded
- `index.csv` ground-truth coordinates equal `dataset.json` exactly — 500/500
- `index.csv` response counts equal the files on disk — 3,446
- every unscored cell is a recorded API failure carrying the source's error string — 554/554,
  and 4,000 − 3,446 = 554
- each `summary.json` recomputes from the response files beside it — 8/8 cells
- each half's mean and count reproduce the source run's own `summary.json` — 16/16 run-halves
- `perceptual_metrics.json` covers all 500 pairs and no SSIM exceeds 1.0
- `results/geolocation_100/` covers exactly the 100-image half of this set
