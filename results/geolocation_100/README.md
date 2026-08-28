# Geolocation, 100 images, iterative verification — VLM reasoning

Every VLM response behind the iterative-verification geolocation experiment, one file per
response. **Reasoning only — no images are included.** The per-image numbers these responses roll
up into are in [`index.csv`](index.csv); the source run's own `evaluation_results.json` is not
shipped. [`examples/results/geolocation/`](../../examples/results/geolocation/) carries two worked
before/after pairs, but from a separate four-image smoke run rather than from this one, so its
numbers are not a subset of the table here.

The run is the loop of `configs/geolocation.yaml` with `max_iter: 3` and the then-current
`val_dist: 5.0`: read the image, remove the street- and city-level cues, **re-read the edited
image**, and stop as soon as the oracle's guess is at least 5 km off — otherwise edit again, up to
three verifications.

> **Produced before the granularity verification change.** This run used the superseded
> `experiment.val_dist: 5.0` stop rule and the earlier attacker prompt, which required the attacker
> to report at least one street-level clue. The current code derives its stop threshold from
> `experiment.mask_level` instead — city-level protection stops at **10 km**, not 5 — and the
> attacker prompt now permits an empty clue list. Both the defense's behaviour and the measuring
> instrument therefore differ from what produced the numbers below, and these responses cannot be
> used to evaluate the residual-cue criterion. See [docs/VERIFICATION.md](../../docs/VERIFICATION.md).

```
results/geolocation_100/
├── MANIFEST.json                          provenance, counts, and the checks below
├── index.csv                              100 rows: ids, ground truth, and every error in one table
├── clue_stats.json                        the run's cue-selection counter (see the caveat below)
├── process/<image_id>/                    part 1 — reasoning during protection
│   ├── round1_full_img.json                 the oracle reading the unprotected image
│   └── iter<N>_full_img.json                verification N: re-reading the image after edit round N
└── evaluation/in_loop_oracle/             part 2 — reading the finished image
    ├── summary.json
    └── responses/<image_id>.json
```

100 images, 344 responses: 244 during protection, 100 at evaluation.

## Part 1 — reasoning during the process

`process/<image_id>/round1_full_img.json` exists for all 100 images. It is the oracle's read of the
**unprotected** Cityscapes frame: which cues it used, a level per cue (country / city / street), and
a latitude-longitude guess with a confidence.

That file is also the raw baseline. Its great-circle error equals the `feature_removed: "none"`
entry in `evaluation_results.json` for all 100 images — checked, zero mismatches. The 65.8 km raw
mean is the mean of these 100 errors.

`iter<N>_full_img.json` is the same oracle re-reading the image after edit round *N*. It does two
jobs at once: its distance from the ground truth is what the stop rule tests, and, if the loop
continues, its cue list is what edit round *N+1* removes. How many an image has is therefore how
many verifications it took:

| Verifications | Images | Meaning |
|---|---|---|
| 1 (`iter1` only) | 73 | one edit round was enough — the oracle was already ≥ 5 km off |
| 2 (`iter1`, `iter2`) | 10 | the first edit left it within 5 km; the second pushed it out |
| 3 (`iter1`…`iter3`) | 17 | ran to `max_iter`; **13 of these were still within 5 km at the end** |

So 87 images met the threshold and 13 exhausted the budget. Nothing is padded: an image that
stopped early simply has no later `iter` files. The figure script (`utils/iter_cdf_paper.py` in the
pipeline repo) carries the last available value forward, which is what the "after 2nd / 3rd" columns
below do too.

## Part 2 — evaluation of the protected image

`evaluation/in_loop_oracle/responses/<image_id>.json` is one more read of the finished image,
`<image_id>_final.png`, with the same prompt and schema. Its error equals the
`feature_removed: "street level clue, city level clue"` entry in the aggregate for all 100 images.

**This is the same oracle that ran inside the loop, not an independent one.** The protected number
is therefore measured by the very model the edits were tuned against; it is a within-loop result,
not a transfer result. No second model scored this image set. (`results/compositional_500/` does
have a second evaluator, but that is a different image set and a different metric — nothing carries
over.)

**The model id is not recorded anywhere.** No response file and no aggregate carries a model field,
and no run log survives. `configs/geolocation.yaml` names `gpt-5`, but that is the default for a
*new* run rather than a record of this one, so the directory is named by role instead of by model.
An independent evaluator, when one is run, goes in a sibling directory named for its recorded id.

### The evaluation and the last verification read the same pixels

For all 100 images the file the final `iter<N>` response read is byte-identical to the file the
evaluation response read — verified by md5 on the source run. The two calls differ only in being two
samples from the same model, and they do not always agree:

- 89 / 100 agree on the "≥ 5 km" criterion.
- On 10 images the loop stopped because it was ≥ 5 km off, yet the second call landed back inside
  5 km; on 1 image the reverse.
- Median absolute gap between the two: 7.1 km.

That is worth knowing before reading the stop rule as a guarantee: `val_dist` is decided on a single
sample, and a single sample of this oracle moves.

## The numbers these files produce

Error is great-circle distance from the Cityscapes GPS ground truth. Higher is better protection.

| | Raw | After 1st | After 2nd | After 3rd | Evaluation |
|---|---|---|---|---|---|
| Mean error (km) | 65.8 | 111.7 | 126.4 | 137.9 | 130.1 |
| Median error (km) | 4.3 | 62.8 | 72.1 | 79.0 | 71.7 |
| Share ≥ 5 km | 50% | 73% | 83% | 87% | 78% |

"After 2nd / 3rd" carry the last available verification forward for images that stopped early, so
all five columns cover the same 100 images. Counting only the images that actually reached them, the
2nd verification covers 27 images (median 2.1 km) and the 3rd covers 17 (median 1.8 km) — those are
the hard images, which is why their medians are low.

Median matters more than mean here: the raw median is 4.3 km, so half the images were already
located to within a city before any editing, and the mean is dragged by a long tail of images the
oracle placed in the wrong country. Editing removed a median of 9.6% of the frame (mean 22.1%).

## `clue_stats.json`, and what its second block is not

Copied unchanged from the run. It counts cues at their most specific level, `selected` (passed the
`mask_level` filter and went to grounding) against `skipped` (grounding found nothing).

Its `round1` block reproduces exactly from the files here: recounting the 100 `round1_full_img.json`
responses gives 191 city-level and 151 street-level cues, matching the recorded 191 / 151. Of those,
173 and 127 were successfully grounded — an 87.7% grounding rate.

Its `round2` block does **not** reproduce. It records 205 selected cues; recounting the
`iter*_full_img.json` responses that led to a further edit gives 162, and no subset of these files
gives 205. `round2` is a counter incremented inside the segmentation stage across every post-round-1
grounding pass, and the run was resumable, so it reflects grounding work rather than the contents of
these responses. Read the `round1` block as a summary of the files in this folder; read `round2` as
a record of the grounding stage only.

## Response shape

Both parts share one schema:

```json
{
  "reasoning": "country / city / street reasoning, in prose",
  "location_estimate": {"latitude": 50.7766, "longitude": 6.0834,
                        "confidence": "High", "region": "Germany, ... , Aachen"},
  "critical_features": {"features": [
    {"object_description": "brown tourist sign on two poles",
     "non_geolocation_element": "brown, rectangular, metal posts",
     "geolocation_element": "readable text 'Aachen', German tourist-sign style",
     "target_concept": "tourist sign",
     "properties": ["country level clue", "city level clue", "street level clue"],
     "region": "across region"}
  ]},
  "image_path": "which image this response read"
}
```

Evaluation responses carry the same four keys, but each feature is reduced to
`geolocation_description` plus `properties` — the evaluator names the residual cue, it does not
describe an object to remove. `image_path` is kept in both, relative to the pipeline run root, so
each response still says which image it read.

## The image set

`index.csv` is the pinned list: 100 Cityscapes training frames, 5 from each of 20 cities, with the
ground-truth coordinates, the number of verifications, every per-stage error, and the mask coverage.
`manifests/` has no geolocation entry — the geolocation manifest is built from a Cityscapes install
(see [`docs/DATA.md`](../../docs/DATA.md)), so this table is what fixes the evaluated set.

## What was checked

Run against the source caches; all ten passed:

- 100 image ids, identical across `process/`, `evaluation/` and the aggregate
- every image directory holds `round1_full_img.json` plus a contiguous `iter1..iterN`
- all 344 responses parse as JSON and carry a non-empty `reasoning` — none empty, none truncated
- `round1` error == `feature_removed: "none"` — 100/100
- evaluation error == `feature_removed: "street level clue, city level clue"` — 100/100
- the stop rule holds: every non-final verification < 5 km, every final one ≥ 5 km unless
  `max_iter` was reached — 100/100
- the last verification and the evaluation read byte-identical images — 100/100
- `clue_stats.json` `round1` recounts exactly; `round2` does not (documented above)
- md5 of every copied file equals its source — 345/345 (the 344 responses plus `clue_stats.json`)

`MANIFEST.json` records the same checks in machine-readable form, along with the source path each
file was copied from.
