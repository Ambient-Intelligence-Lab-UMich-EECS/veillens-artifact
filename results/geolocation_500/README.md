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
    │   ├── gemini-3.1-pro-preview/                                                    500
    │   └── gemini-2.5-flash/                                                          500
    ├── gpt-image/                        one gpt-image-2 edit per frame
    │   ├── gpt-5.2/                                                                   500
    │   ├── gemini-3-flash-preview/                                                    500
    │   ├── gemini-3.1-pro-preview/                                                    500
    │   └── gemini-2.5-flash/                                                          500
    ├── veillens/PENDING.json             the method itself — never run on these 500
    ├── rand-noise/PENDING.json
    ├── ocr-blur/PENDING.json
    ├── mm-pgd/PENDING.json
    └── geoshield/PENDING.json
```

500 images, **4,000 responses** — the matrix is complete: every image carries all eight.

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
| **raw**, Gemini 3.1 Pro | 46.6 | 58.6 | 71.4 | 72.8 | 77.2 | 85.2 | 500 | 39.0 | 0.61 km |
| **raw**, Gemini 2.5 Flash | 15.0 | 29.0 | 45.8 | 47.2 | 53.4 | 72.4 | 500 | 86.3 | 18.19 km |
| `gpt-image`, GPT-5.2 | 4.8 | 8.6 | 16.6 | 17.0 | 19.6 | 40.2 | 500 | 197.5 | 162.83 km |
| `gpt-image`, Gemini 3 Flash | 14.2 | 22.2 | 32.6 | 33.6 | 37.2 | 54.8 | 500 | 132.1 | 71.25 km |
| `gpt-image`, Gemini 3.1 Pro | 15.4 | 21.8 | 31.6 | 32.4 | 34.8 | 50.6 | 500 | 140.5 | 89.70 km |
| `gpt-image`, Gemini 2.5 Flash | 7.6 | 15.8 | 26.6 | 27.6 | 31.2 | 54.4 | 500 | 153.9 | 72.03 km |

Every row is *n* = 500 and every row is the same 500 images.

The raw rows are the finding to read first: **Gemini 3 Flash puts the median image within 780 m of
its true position**, and Gemini 3.1 Pro within 610 m, on ordinary dashcam frames with no landmark
requirement. GPT-5.2 and Gemini 2.5 Flash are far weaker attackers on this set. Any claim about a
defence is a claim relative to whichever of these attackers is being run.

### Paired, on the images each evaluator scored in both conditions

| Evaluator | n | raw median | edited median | < 25 km, raw → edited | further | closer | unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.2 | 500 | 25.14 km | 162.83 km | 50.0% → 19.6% | 331 | 130 | 39 |
| Gemini 3 Flash | 500 | 0.78 km | 71.25 km | 72.6% → 37.2% | 379 | 121 | 0 |
| Gemini 3.1 Pro | 500 | 0.61 km | 89.70 km | 77.2% → 34.8% | 379 | 101 | 20 |
| Gemini 2.5 Flash | 500 | 18.19 km | 72.03 km | 53.4% → 31.2% | 316 | 183 | 1 |

The edit helps against every evaluator, and it is nowhere near a fix. Gemini 3 Flash still places
**a third of the edited images inside 25 km**, and on 121 of 500 images it located the edited photo
*better* than the original. "Unchanged" is the model returning the same rounded coordinate for both
pictures — a city-centroid guess the edit never disturbed; GPT-5.2 does this on 39 images.

Median is the honest column here. The means are dragged by a long tail of images placed in the wrong
country, and a defence that moves one guess from Aachen to Portugal buys a large mean improvement
for no privacy that a 25 km threshold would not already have granted.

## How the Gemini 3.1 Pro cells were filled

Both Gemini 3.1 Pro cells were incomplete for most of this set's life, and the reason is worth
recording because it shapes what the archives look like. The model has a
**250-requests-per-model-per-day cap**. Its 1,000 calls do not fit in a day, so the first job
exhausted the quota part way through the `gpt-image` set and never reached `raw` at all. Every
blocked call returned HTTP 429 with a null estimate, which the runner records as a null error rather
than dropping the row, so alignment survived and the gap was resumable.

It was closed by scheduling the remainder as further chunks on later days, each built against the
ids still missing so no call was ever paid for twice:

| chunk | date | new `gpt-image` answers | new `raw` answers |
|---|---|---:|---:|
| `gpt_image_100_*` / `_resume_b_*` | 2026-08-13 | 100 | 100 |
| `gpt_image_500_new400_c_*` | 2026-08-20 | 248 | 0 |
| `gpt_image_500_new400_k2_*` | 2026-08-24 | 147 | 73 |
| `gpt_image_500_new400_c2_*` | 2026-08-24 | 3 | 245 |
| `gpt_image_500_new400_d_*` | 2026-08-28 | 2 | 82 |
| | | **500** | **500** |

Answers are unioned by `image_id` with the first non-null winning, so a later chunk that hit the cap
again could not erase an earlier success. The other gap was smaller and ordinary: two API failures
for Gemini 3 Flash on the edited images (`aachen_000083_000019`, `weimar_000000_000019`), retried in
`gpt_image_500_new400_e_*` and both scored.

**The Pro numbers moved a long way when the sample grew.** The 2026-08-20 snapshot published its
edited row on 348 images (median 139.42 km) and its paired row on the 100-image half (median
129.65 km, 29.0% inside 25 km). On all 500 those become **89.70 km and 34.8%**. The raw row moved
much less: median 0.61 km then and now.

The direction of that shift is the 100-image half being harder on the edited side — its edited
median is 129.65 km against 76.48 km on the other 400 — but that difference is **not clearly beyond
sampling noise** (Mann-Whitney *p* = 0.065), and both halves are balanced the same way, 5 images per
city and 20 per city across the same 20 cities. Gemini 3 Flash shows the same direction more
strongly (106.79 vs 68.34 km, *p* = 0.019); GPT-5.2 and Gemini 2.5 Flash show essentially nothing
(*p* = 0.51, 0.43). So this is offered as a caution, not a finding about the images.

The safe reading is the plain one: a 100- or 348-image cell was never interchangeable with a
500-image one, and the *n* column is not decoration. Any row here with *n* < 500 should not be read
against the rows beside it — which, now, none are.

`index.csv` now has no blank error cells. No response file was dropped — checked.

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
`haversine(ground truth, location_estimate)` for every one of the 4,000 responses reproduces the
recorded `prediction_errors_km` **exactly** — zero difference, not merely small — at R = 6371.0 km.
A misalignment would have to survive 4,000 exact coincidences.

Answers are unioned by `image_id` across every matching run, any non-null answer winning, which is
the same rule as `benchmark/analysis/merge_gpt_image_500.py`. A 10-image smoke run
(`gpt_image_10_20260813_103949`) is excluded: it is a subset of the 100 that the 100-image run
already covers.

## What was checked

Everything here is built by
[`benchmark/analysis/build_geolocation_500_artifact.py`](../../benchmark/analysis/build_geolocation_500_artifact.py)
in the VeilLens repository, which re-derives every file from the run archives and refuses to write
if any check below fails. Re-running it on unchanged archives reproduces this directory byte for
byte; that is how the backfill was folded in without disturbing what was already published.

All 13 passed; `MANIFEST.json` records them in machine-readable form.

- the two halves partition the 500-image set — 100 + 400, disjoint, union equals `dataset.json`
- every emitted response parses as JSON, carries a non-empty `reasoning`, and has exactly the keys
  of `prompts/eval_schema.json` — 4,000/4,000
- emitted bodies are deep-equal to the source objects — 4,000, zero differences
- recomputed `haversine(gt, prediction)` equals the recorded error — 4,000/4,000, **exactly**
- `index.csv` lists the 500 once, in `dataset.json` order
- every `index.csv` error cell equals the source's `prediction_errors_km` exactly — 4,000 cells,
  written at full round-trip precision rather than rounded
- `index.csv` ground-truth coordinates equal `dataset.json` exactly — 500/500
- `index.csv` response counts equal the files on disk — 4,000
- every unscored cell is a recorded API failure carrying the source's error string — 0 unscored,
  and 4,000 − 4,000 = 0
- each `summary.json` recomputes from the response files beside it — 8/8 cells
- each run's mean and count reproduce that run's own `summary.json` — 23/23 run-cells
- `perceptual_metrics.json` covers all 500 pairs and no SSIM exceeds 1.0
- `results/geolocation_100/` covers exactly the 100-image half of this set
