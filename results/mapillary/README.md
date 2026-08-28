# Mapillary holdout, 217 images — VLM reasoning

Every VLM response that still exists behind the MAPILLARY geolocation experiment, one file per
response. **Reasoning only — no images are included.**

This folder is smaller than its siblings, and the reason is not editorial. Most of the run's caches
have been deleted from the filesystem. What is here is what survives; the gaps are enumerated rather
than smoothed over, because the gaps are the finding.

```
results/mapillary/
├── MANIFEST.json                          provenance, counts, and the checks below
├── README.md
├── index.csv                              217 rows: ids, ground truth, and every error that survives
├── recover_image_ids.py                   re-derives which image each response belongs to
└── evaluation/                            reads of the finished protected image
    ├── claude-haiku-4-5/                    summary.json + responses/<image_id>.json   156
    ├── gemini-3-flash-preview/              summary.json + responses/<image_id>.json   157
    └── gpt-5/                               summary.json + responses/<image_id>.json     2
```

217 images, **315 responses**. There is no `process/` part.

## The image set

`pipeline2/mapillary/mega/selected.json` pins 217 Mapillary street-level images over **18 cities**,
reproduced in `index.csv`. It is a filtered subset of a 341-image pull: `pipeline2/utils/selection.py`
keeps only images a raw attacker had already located within a distance threshold, so this is a
hard-subset holdout, not a uniform sample.

The 217 were protected in two runs that partition the set exactly:

| Half | n | Protection run | Why |
|---|---:|---|---|
| `protected_158` | 158 | `pipeline2/result/mega_city_street` | images whose protected final existed when the evaluation manifest was built |
| `protected_59` | 59 | `pipeline2/result/mega_city_street2` | the remainder, protected in a resume run |

Disjoint, and their union is exactly `selected.json` — which is why `accuracy_stats.txt` has a 217
denominator while the surviving evaluation run reports 158.

**This is not the image set the paper describes.** §7.1 specifies a 200-image holdout, 20 cities,
10 images per city. Nothing on the filesystem has that shape: it is 217 images over 18 cities, with
between 1 and 20 images per city (Ho Chi Minh City, New York, Iztapalapa and Soweto have 20 each;
Kanpur has 1).

## What survives, and what does not

| Part | Expected | Present | Where it went |
|---|---:|---:|---|
| Process reasoning (`round1_full_img.json`) | 217 | **0** | `original_cache/` still holds one directory per image; every one is empty |
| GPT-5, in-loop oracle (`eval_cache`) | 217 | **2** | 215 files deleted |
| Claude Haiku 4.5 | 217 | **156** | 158-half retained; 2 API failures; the 59-half's `results.json` deleted |
| Gemini 3 Flash (preview) | 217 | **157** | 158-half retained; 1 API failure; the 59-half's `results.json` deleted |
| GPT-5.2 | 217 | **0** | ran on all 217; every per-image `results.json` deleted |
| Gemini 2.5 Flash | 217 | **0** | ran on all 217; every per-image `results.json` deleted |
| Gemini 3 Pro | — | **0** | never run on this image set |

**No image in this holdout has all four evaluators, and none has any process reasoning.** 154 images
have two responses, 4 have one, 1 has three, and 58 have none. The deleted evaluators are not gone
without trace: their aggregates survive in the two `comparison/` directories, and the reconciliation
below uses them.

The consequence worth stating plainly: **the raw-image baseline for this holdout barely exists.**
`original_cache` was where the attacker's read of the *unprotected* photograph lived, and it is empty
for all 217. What is left of a raw baseline is coordinates only — no reasoning — for the 59-image
half, in `evaluation_results.json`. For the 158-image half there is none at all.

## The evaluation run's `raw/` directory holds protected images

The evaluation run names its image set `raw`, and its `experiment_config.yaml` describes it as
`Original unmodified images from dataset (baseline)`. Both are a stale template label — the same
config file also points at a different result directory than the one it ran in.

`pipeline2/utils/eval_dataset.py` builds the evaluated manifest by rewriting every `image_path` to
`intermediate/<image_id>/final/<image_id>_final.png`, keeping only images whose protected final
exists. So what these four models scored is the VeilLens output. `pipeline2/utils/accuracy.py`, which
built `accuracy_stats.txt` from these same files, labels every row `Processed` for the same reason.

**None of these four models has a raw-image score on this holdout.** Any raw-versus-protected
comparison drawn from this run is comparing a GPT-5 raw number against a different model's protected
number.

## Which image each response belongs to

`results.json` stores its responses as a positional array with no image id, and the manifest that
fixed the order — `selected_eval.json` — does not exist anywhere on the filesystem. The mapping in
this folder was recovered, not read.

For response *i*, the recorded error pins the ground truth: the image is whichever candidate `g` in
`selected.json` satisfies

```
| haversine(g, responses[i].location_estimate) − prediction_errors_km[i] | < 1e-9 km     (R = 6371.0)
```

That is unique for **152 of the 158** slots. The remaining 6 follow by force — the resolved slots are
strictly increasing in `selected.json` order, and each unresolved run of *k* slots sits in a window of
exactly *k* candidates.

The result is confirmed by evidence that was never used to build it. The run's own `failures` list
names the images whose API call failed, and those ids land exactly on the slots whose response is
null: 2/2 for Haiku, 1/1 for Gemini 3 Flash.

`recover_image_ids.py` reproduces the mapping and re-runs both checks against a `pipeline2/` tree.
Six ids rest on the ordering argument alone rather than a distance match; `MANIFEST.json` names them.

## The numbers these files produce

Percentage of images located within each threshold. Higher is *worse* protection.

| | 0.5 km | 1 km | 5 km | 10 km | 25 km | 50 km | n | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5, protected | 0.6 | 3.8 | 7.1 | 13.5 | 22.4 | 23.1 | 156 | 725.3 km |
| Gemini 3F, protected | 9.6 | 12.7 | 24.8 | 35.0 | 56.7 | 63.1 | 157 | 20.4 km |
| GPT-5, **raw** | 15.3 | 27.1 | 45.8 | 61.0 | 84.7 | 84.7 | 59 | 5.4 km |
| GPT-5, protected | 5.1 | 16.9 | 27.1 | 32.2 | 62.7 | 62.7 | 59 | 15.1 km |

The two evaluator rows cover the 158-image half; the two GPT-5 rows cover the 59-image half and come
from coordinates, not from responses. **No row here is a raw/protected pair on the same images by the
same model** — the only such pair in the whole holdout is the GPT-5 one, on 59 images, without
reasoning.

## Reconciling with `accuracy_stats.txt`

`pipeline2/result/mega_city_street/accuracy_stats.txt` reports over all 217 images. Its per-row
denominators are 217, less each model's API failures — 215 for Haiku, 216 for Gemini 3 Flash. Adding
the surviving 158-half count to the 59-half aggregate reproduces its 25 km column exactly for all
four evaluators:

| Row | 158-half | 59-half | total | recomputed | in the file |
|---|---:|---:|---:|---:|---:|
| Processed Haiku-4.5 | 35 | 8 | 43 / 215 | 20.0% | 20.0% |
| Processed Gemini 2.5 Flash | 91 | 40 | 131 / 217 | 60.4% | 60.4% |
| Processed Gemini 3 Flash | 89 | 42 | 131 / 216 | 60.6% | 60.6% |
| Processed GPT-5.2 | 59 | 24 | 83 / 217 | 38.2% | 38.2% |

Its other two rows, `Original GPT-5` and `Processed GPT-5`, are computed over 217 images, so a
217-entry `evaluation_results.json` existed when the table was written. Only the 59-entry one
survives, and `mega_city_street/evaluation_results/` is empty. **Those two rows cannot be recomputed
from anything in this repository** — they are traceable to a number, not to records.

## Response shape

The two independent evaluators return two keys and nothing else:

```json
{
  "reasoning": "country / city / street reasoning, in prose",
  "location_estimate": {"latitude": -6.2195, "longitude": 106.8152,
                        "confidence": "High", "region": "Jakarta, Indonesia"}
}
```

No `critical_features`, and no `image_path` — which is why the image id had to be recovered. The two
`gpt-5` responses use the fuller in-loop schema, adding `critical_features` (each feature reduced to
`geolocation_description` plus `properties`) and `image_path`.

## What was checked

All 15 passed; `MANIFEST.json` records them in machine-readable form.

- every response slot resolves to exactly one image id — 158/158, no conflicts between evaluators
- recovered ids are strictly increasing in `selected.json` order — 158/158
- the run's declared API failures land on the recovered ids — 2/2 Haiku, 1/1 Gemini 3 Flash
- recomputed `haversine(gt, prediction)` equals the recorded error — 156/156 and 157/157, to < 1e-9 km
- the two halves partition the 217-image holdout — 158 + 59, no overlap, union equals `selected.json`
- `mega_city_street2`'s `evaluation_results.json` covers exactly the 59-image half
- every emitted response parses as JSON with a non-empty `reasoning` — 315/315
- emitted bodies are deep-equal to the source objects — 315/315
- the single retained GPT-5 response reproduces its coordinate record exactly (8.177084 km)
- `accuracy_stats.txt`'s 25 km column recomputes from the two halves for all four evaluators
- `index.csv` lists the holdout once, in `selected.json` order, and its response counts match the
  files on disk — 315
