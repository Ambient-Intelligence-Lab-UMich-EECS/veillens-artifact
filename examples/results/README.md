# Worked examples — two images per privacy setting

Six images in total, two per setting, copied unmodified out of real runs. Enough to see what the
loop emits and to exercise both offline commands from a fresh clone — **no API key, no GPU, no
dataset download.**

```
examples/results/<setting>/
├── evaluation_results/evaluation_results.json   the aggregate, two records
├── eval_cache/<image>_eval.json                 the attacker's raw reply, per image
└── images/<image>_{original,protected}.jpg      what went in, and what came out
```

Both commands take the directory:

```bash
veillens-score  examples/results/geolocation
veillens-verify examples/results/geolocation
```

`veillens-score` reports the distribution — how far off the attacker was.
`veillens-verify` reports the verdict — whether the cue granularities the run promised to remove
are actually gone. It finds `eval_cache/` on its own, so neither command needs a flag.

**These six images are a demonstration of the output format and of what the edit looks like. They
are not evidence of effect size** — two images cannot be. Every reported number comes from the
full runs archived in [`results/`](../../results/), which carry one file per VLM response for all
500, 100 and 454 images respectively.

## What each sample shows

| Setting | Images | Source run | Protected image is |
|---|---|---|---|
| `geolocation` | `aachen_000000_000019`, `erfurt_000000_000019` | 4-image smoke run over Cityscapes | SDXL inpainting — billboards and signage repainted as generic artwork |
| `compositional` | `2017_15522226`, `2017_16257673` | re-run on the current pipeline (the archived run used a different one — see below) | SDXL inpainting in the config's 1960s-cartoon style |
| `social_relation` | `6917`, `9545` | 5-image smoke run over PISC | face-band inpainting inside the dataset's person boxes, which the run draws as `P1`/`P2` |

### geolocation

Both images start correctly localised and end far away. `erfurt_000000_000019` is the clearer of
the two: German-language billboards (`Grüne Tage Thüringen`) and street furniture are repainted,
and the attacker moves from 1.5 km off to 318 km off. City-level protection asks for ≥ 10 km, so
both pass the metric criterion.

Both fail the *residual* criterion — the attacker's own report still names street- and city-level
cues. `veillens-verify` prints that disagreement rather than hiding it, along with the reason it
may be an artefact of the attacker prompt. See [VERIFICATION.md](../../docs/VERIFICATION.md).

### compositional

Both panelists' faces are repainted in the style `configs/compositional.yaml` asks for
(`positive_prompt_prefix: "Mid century artstyle, cartoonistic artstyle, 1960s cartoon style"`), and
the desk nameplates are covered. Risk score falls 0.83 → 0.62 on `2017_15522226` (22% of the frame
altered) and 0.72 → 0.58 on `2017_16257673` (23%) — a 22.6% mean reduction over the pair.

**These two images do not come from the run behind
[`results/compositional_500/`](../../results/compositional_500/), and they should not be read as a
sample of it.** That archived run was produced by an earlier pipeline that differed in two ways
that both matter:

* Its protected image was a **flat grey composite**, not an inpainting. In the loop as it ships,
  the grey composite is a diagnostic written to `intermediate/<image>/round1/` and `final/` holds
  the SDXL result; in the archived run the grey composite *was* `final/`, and that is the image
  `MANIFEST.json` records as the `scored_image` for both of its attackers.
* It **downscaled every input** to a 1024px long side. The current pipeline does that too, but
  only because the compositional spec now declares `max_input_dim: 1024` — see
  [USAGE.md](../../docs/USAGE.md#input-resolution) for why that cap exists.

Re-running these two images on the current pipeline gives a **much smaller reduction than the
archived numbers**: 22.6% here against 91.3% for the same two images in the archived run, which
masked no more of the frame (20.1% and 9.4%). The mechanism is visible in the images. Grey fill
destroys the region; inpainting substitutes something plausible, so a repainted face still reads as
a face, and the conference banner and scene the risk score also draws on are untouched either way.
Both images now fail the residual criterion, with severity-1 cues surviving.

**Two images cannot settle what the same comparison would do across all 500** — that would need the
full set re-run, which has not been done. But the headline compositional figure in
`results/compositional_500/` (mean risk 0.3138 → 0.1611, a 48.6% relative reduction) was measured
on grey-masked images, and it should not be read as a measurement of the inpainting pipeline in
this repository.

### social_relation — a negative result

**On these two images the defense does not work, and the sample is kept that way on purpose.**
The faces are anonymized, and the attacker still labels `9545` a couple and still recovers
`6917`'s pairs — better after the edit than before. Posture, proximity and an arm around a
shoulder survive a face edit, and those are what the relationship reads from.

That is a property of this path, not of the sample: `configs/social_relation.yaml` runs face
anonymization at `max_iter: 0`. The defense that the paper reports for this setting is a different
mechanism — **annotation withholding**, which removes a subject's box so the attacker is never
asked about the pair — and it is measured over 454 images in
[`results/social_relationship_454/`](../../results/social_relationship_454/), together with its own
largest caveat (71 images the defense silently declined to protect).

## Record shape

Each file is keyed by `image_id` and holds a `feature_results` list. Its first entry is always
`feature_removed: "none"` — the unprotected image — and each later entry is the state after a set
of cues was removed. That is what makes one file self-contained: the baseline and the protected
result come from the same attacker in the same run, so the reduction is not a comparison across
two separately-configured jobs.

`mask_pixels` and `mask_percentage` record how much of the frame each step altered, which is the
denominator for any "how much did it cost to remove" question.

## Attribution

The compositional records carry no attacker model field, and nothing external pins it down — the
format has no slot for it. Anyone citing a number from these files should confirm the model
separately rather than assume it matches `configs/*.yaml`, whose `evaluation.model` is only the
default for a *new* run. `results/*/MANIFEST.json` records the attacker for the archived runs.

## Images

The originals are frames from Cityscapes, VISPR and PISC, downscaled and re-encoded as JPEG for
size. The full datasets are not redistributed here; see [DATA.md](../../docs/DATA.md) for how to
obtain them.
