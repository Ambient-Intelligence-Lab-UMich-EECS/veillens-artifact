# Compositional privacy (CPRT), 500 images — VLM reasoning

Every VLM response behind the compositional-privacy experiment, one file per response.
**Reasoning only — no images are included.** The image set is the one named in
[`manifests/compositional_500.json`](../../manifests/compositional_500.json); the per-image scores
those responses roll up into are in [`index.csv`](index.csv). Two of the 500 images appear as a
worked before/after pair in
[`examples/results/compositional/`](../../examples/results/compositional/).

```
results/compositional_500/
├── MANIFEST.json                          provenance, counts, and the checks below
├── index.csv                              500 rows: every score in one table
├── process/<image_id>/                    part 1 — reasoning during protection
│   ├── round1_full_img.json                 the oracle reading the unprotected image
│   └── round2_<k>.json                      re-reading crops after round 1's edits (0, 2 or 4)
└── evaluation/<model>/                    part 2 — scoring the protected image
    ├── summary.json
    └── responses/<image_id>.json
```

## Part 1 — reasoning during the process

`process/<image_id>/round1_full_img.json` exists for all 500 images. It is the oracle's first
read of the **unprotected** input: what it believes is inferable, which visual cues carry that
inference, a severity level per cue, and a risk score.

That file is also the raw baseline. Its `personal_info_privacy_score.score` equals the
`feature_removed: "none"` entry in the source run's own `evaluation_results.json` for all 500
images — checked against that run, zero mismatches; the `raw_score` column of `index.csv` is that
same number. The 0.3138 raw mean is the mean of these 500 numbers.

`round2_<k>.json` is the same oracle re-reading crops of the image after round 1's edits landed.
The count varies with how many rounds an image needed: 42 images have none, 421 have two, 37 have
four — 1,490 process responses in total.

## Part 2 — evaluation of the protected image

Each evaluator scored the same final protected image, `<image_id>_final.png`, with the same
prompt and schema.

| Evaluator | Directory | Responses | Model id in the files |
|---|---|---|---|
| GPT-5.2 | `evaluation/gpt-5.2/` | 500 / 500 | *(none recorded)* |
| Gemini 3F | `evaluation/gemini-3.1-flash-lite/` | 500 / 500 | `gemini-3.1-flash-lite` |
| Gemini 3P | — | **0** | run does not exist |
| Gemini 2.F | — | **0** | run does not exist |

**Two of the four evaluators have no data.** No Gemini 3 Pro or Gemini 2.5 Flash run against this
image set exists on the filesystem this folder was built from. Both models were run for the
*geolocation* experiment, which is a different image set and a different metric; nothing from
there applies here. Until those two runs are performed, this folder holds two evaluators.

**GPT-5.2 is an attribution, not a record.** The response files carry no model field, and neither
does the source run's aggregate. `configs/compositional.yaml` names an `evaluation.model`, but
that is the default for a *new* run rather than a record of this one. `gemini-3.1-flash-lite` is
different — the second attacker's aggregate names it in a top-level `model` field.

## Response shape

Both parts share one schema:

```json
{
  "reasoning": "why this score",
  "personal_info_privacy_score": {"score": 0.24, "confidence": "Medium"},
  "critical_features": {"features": [
    {"object_description": "...", "privacy_element": "...", "Anonymized_element": "...",
     "target_concept": "...", "severity_level": 3, "attributes": ["..."]}
  ]}
}
```

Process responses add `image_path` (which image was read) and a `region` field per feature.
`severity_level` runs 1 (biometric / direct identifier) to 4 (weak contextual cue); the protection
loop removes cues at severity ≤ 2, which is what `feature_removed: "severity<=2"` means in the
aggregate files.

## The numbers these files produce

| | Raw | GPT-5.2 | Gemini 3F |
|---|---|---|---|
| Mean risk, all 500 | 0.3138 | 0.1611 | 0.1641 |
| Mean risk, 273 high-concern (GT ≥ 0.5) | 0.5224 | 0.2456 | 0.2456 |
| Relative reduction | — | 48.6% | 47.7% |

**Only GPT-5.2 measured the raw images.** Every Gemini response scores a `final/` image, so the
Gemini reduction is its protected mean against *GPT-5.2's* raw baseline, not a before/after pair
it measured itself. The two are close enough to read as agreement, but they are not the same
quantity, and a Gemini raw pass would be needed to make them so.

## What was checked

Run against the source caches before copying; all five passed with zero exceptions:

- the 500 image ids match `manifests/compositional_500.json` exactly
- `round1_full_img.json` score == `feature_removed: "none"` — 500/500
- `evaluation/gpt-5.2` score == `feature_removed: "severity<=2"` — 500/500
- `evaluation/gemini-3.1-flash-lite` score == `predicted_score` — 500/500
- `ground_truth_score` agrees between the two evaluators' aggregates — 500/500 at 1e-9

No response file is missing, empty, or unparseable. `MANIFEST.json` records the same checks in
machine-readable form, along with the source path each file was copied from.
