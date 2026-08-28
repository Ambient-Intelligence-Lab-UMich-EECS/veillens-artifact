# Bundled run records

Three JSON files from finished runs, so that `veillens-score` can be exercised with no API key,
no GPU and no images. They are the pipeline's own output, copied unmodified.

| File | Rows | What it is |
|---|---|---|
| `geolocation/evaluation_results.json` | 100 | Per-image geolocation records: ground-truth coordinates, and one entry per cue set removed, each with the attacker's guess and its great-circle error |
| `compositional/run_evaluation_results.json` | 500 | Per-image compositional records over the CPRT-Bench subset in `manifests/compositional_500.json`: ground-truth risk score, and the attacker's score before and after protection |
| `compositional/evaluation_results.json` | 500 | A **second, independent attacker** re-scoring the same 500 protected images. Protected-only — it carries no before/after pair, so it measures whether the protection holds against a model other than the one in the loop |

```bash
veillens-score examples/results/geolocation/evaluation_results.json
veillens-score examples/results/compositional/run_evaluation_results.json --setting compositional
```

The second reproduces the headline compositional number: mean risk 0.3138 → 0.1611, a 48.6%
relative reduction.

## Record shape

Both per-image files are keyed by `image_id` and hold a `feature_results` list. Its first entry is
always `feature_removed: "none"` — the unprotected image — and each later entry is the state after
another set of cues was removed. That is what makes a single file self-contained: the baseline and
the protected result come from the same attacker, in the same run, so the reduction is not a
comparison across two separately-configured jobs.

`mask_pixels` and `mask_percentage` record how much of the frame each step altered, which is the
denominator for any "how much did it cost to remove" question.

## Attribution, and what is missing from it

`compositional/evaluation_results.json` names its attacker in a top-level `model` field:
`gemini-3.1-flash-lite`.

**The other two files record no attacker at all.** The format has no field for it, and nothing
external pins it down — so which model produced `geolocation/evaluation_results.json` and
`compositional/run_evaluation_results.json` cannot be established from this repository. Anyone
citing a number from them should confirm the model separately rather than assume it matches
`configs/*.yaml`, whose `evaluation.model` is only the default for a *new* run.

These are the aggregates. The individual VLM responses they roll up from — the oracle's reasoning during protection, and each evaluator's reasoning about the protected image — are one folder per experiment under `results/`:
[`results/compositional_500/`](../../results/compositional_500/) for the two compositional files, and
[`results/geolocation_100/`](../../results/geolocation_100/) for `geolocation/evaluation_results.json`, whose
344 responses also record what the oracle saw at each verification of the three-iteration loop.
