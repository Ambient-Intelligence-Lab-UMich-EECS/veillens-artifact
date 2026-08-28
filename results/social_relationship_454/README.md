# Social relationship inference (PISC), 454 images — VLM reasoning

Every VLM response behind the social-relationship experiment, one file per response.
**Reasoning only — no images are included.** The image set is the 454-image PISC subset pinned in
`index.csv`; the source run is `pipeline2/data_sc`.

The attack is pair-level relationship classification. Each subject in an image gets a numbered
box, the attacker is asked to label every pair, and accuracy is measured against the PISC ground
truth. The defense is **annotation withholding**: no pixels are altered, but subjects the pipeline
believes carry a relationship simply do not get a box, so the attacker is never asked about them.

```
results/social_relationship_454/
├── MANIFEST.json                          provenance, counts, and the checks below
├── index.csv                              454 rows: per-image outcome and every attacker's score
├── defense_abstention.json                what the defense did to each image, and why
├── process/<image_id>/
│   └── round1_full_img.json               part 1 — gpt-5 reading the unprotected image
└── evaluation/<condition>/<model>/        part 2 — the attack
    ├── summary.json
    └── responses/<image_id>.json
```

`evaluation/` carries an extra `raw` / `defended` level that `results/compositional_500/` and
`results/geolocation_100/` do not have. Those experiments measure their raw baseline inside
`process/`; here the raw baseline is a **separate attacker run against a separate image**
(all subjects boxed, rather than some), so it cannot live under `process/`.

904 files, 897 of them responses: 454 during protection, 361 raw attack, 82 defended attack.

## Labels

| | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| six-way | friends | family-members | couple | professional | commercial | no-relationship |
| three-way | intimate | intimate | intimate | non-intimate | non-intimate | no-relation |

The three-way grouping is the PISC standard coarse split. Every three-way number in this folder is
recomputed from the same response files as the six-way number, not read from a separate run.

## Part 1 — reasoning during the process

`process/<image_id>/round1_full_img.json` exists for all 454 images. It is gpt-5 reading the
**unprotected** image, and it does two jobs: it is the gpt-5 raw baseline, and its `predictions`
field is the input `defense.py` consumes to decide which subjects to withhold.

**This stage runs its own person detector rather than using ground-truth boxes.** That is the one
structural difference between it and the two `attack.py` runs, and it matters for every number:
detection misses subjects, so only **1109 of the 1518** ground-truth pairs (73.1%) ever received an
answer. The response files record the gap directly in `missed_pairs`, `subjects_matched` and
`subjects_unmatched`.

| gpt-5, denominator | pairs | six-way | three-way |
|---|---|---|---|
| all ground-truth pairs | 1518 | 0.5323 | 0.5718 |
| answered pairs only | 1109 | 0.7286 | 0.7827 |

Both are correct; they answer different questions. The 0.5323 figure is what
`pred_output/.../metrics.json` reports. **Neither is directly comparable to the two `attack.py`
runs below**, which are handed ground-truth boxes and answer essentially every pair — a detector's
recall is folded into the first number and excluded from the second, and no `attack.py` run carries
either penalty.

52 responses have an empty `raw_response` and an empty `predictions`. They are not truncated files;
the model returned nothing parseable for those images, and all 52 fall into the abstention bucket
described next.

## The defense, and what it did to the image set

`defense.py` was replayed over all 454 prediction responses for this folder; the replay reproduces
`def_img/dataset.json` exactly — same 41 images, same surviving-pair labels. The 454 images split
four ways, recorded in `defense_abstention.json` and as `defense_outcome` in `index.csv`:

| outcome | images | GT pairs | relationship pairs | anonymized? |
|---|---|---|---|---|
| `protected` | 41 | 396 | 182 (46%) | yes — image written, 80 pairs left answerable |
| `skip_all_relationship` | 320 | 824 | 776 (94%) | yes — everyone would be removed |
| `skip_le1_remaining` | 22 | 153 | 98 (64%) | yes — ≤1 subject left, no pair can form |
| `skip_no_relationship_subject` | **71** | **145** | 102 (70%) | **no — image never written** |
| total | 454 | 1518 | 1158 | |

The first three are genuine protection. An image where every subject is relationship-bearing has
nobody left to box, so the pipeline writes nothing and there is nothing left to attack — skipping
it is the defense succeeding, and those pairs are correctly credited as protected.

**The fourth bucket is not.** Those 71 images reach
`if not rel_subjects: return ..., None` — gpt-5 predicted no pair with a label below 6 (52 images
because it returned nothing at all, 19 because it labelled every pair `no-relationship`), so the
set of subjects to withhold came back empty and the function returned *before writing an image*.
No `def_img` file exists for any of them. The original stays fully annotated and fully attackable.

Their ground truth is not "no relationship" — 102 of their 145 pairs (70%) carry a real
relationship, and 61 of the 71 images are all-relationship. The raw attackers confirm what that is
worth: on exactly these images they score **76.5%** six-way (flash-lite, 68 pairs) and **72.2%**
(pro-preview, 36 pairs). This is a detector-recall failure surfacing as an abstention, and it is
the single largest caveat in this folder.

## Part 2 — the attack

Same script (`attack.py`), same prompt, same schema in both conditions; only the image differs.

| condition | image read | boxes drawn |
|---|---|---|
| `raw` | `annotated/<image_id>.jpg` | every subject |
| `defended` | `def_img/<image_id>.jpg` | only subjects **not** predicted to be in a relationship |

| model | raw | defended |
|---|---|---|
| gemini-3.1-flash-lite | 200 images, 646 pairs | 41 images |
| gemini-3.1-pro-preview | 161 images, 528 pairs | — run does not exist |
| gpt-5.2 | — run does not exist | 41 images |
| gpt-5 | *(process only — different harness, see above)* | — |

**Four distinct models appear here, but no condition was run with all four, and only
gemini-3.1-flash-lite has both.** Every before/after statement in this folder therefore rests on
that one model. Model ids are recorded — the source run directories and `attack.py`'s `MODEL_MAP`
both name them — unlike the other two experiments in `results/`, where attribution is the run
author's.

The raw runs also cover **subsets** of the 454 images (200 and 161), and not the same subset;
`summary.json` in each raw directory breaks its coverage down by defense outcome. Five
pro-preview responses are empty (13 pairs), counted as wrong.

## The numbers these files produce

Raw, each over its own coverage:

| attacker | pairs | six-way | three-way |
|---|---|---|---|
| gemini-3.1-flash-lite | 646 | 0.6827 | 0.7260 |
| gemini-3.1-pro-preview | 528 | 0.7311 | 0.7784 |
| gpt-5 (answered pairs) | 1109 | 0.7286 | 0.7827 |

Defended. The attacker answered 80 surviving pairs on the 41 protected images and got 50 (flash-lite)
and 65 (gpt-5.2) right six-way. Those counts are fixed; only the denominator is a choice:

| denominator | what it assumes | flash-lite 6w / 3w | gpt-5.2 6w / 3w |
|---|---|---|---|
| **80** surviving pairs | conditions on the pairs the defense left answerable | 0.6250 / — | 0.8125 / — |
| **1518** all GT pairs (A) | every unanswered pair is protected | 0.0329 / 0.0349 | 0.0428 / 0.0455 |
| **1373** excluding the 71 (B) | as A, minus the images never anonymized | 0.0364 / 0.0386 | 0.0473 / 0.0503 |

**The 80-pair denominator is the one in the source `metrics.json`, and it is not a defense metric.**
It reports 62.5% and 81.25% — high numbers for a working defense — because 62 of the 80 surviving
pairs are `no-relationship` by construction: the defense removed the relationship-bearing subjects,
so what it leaves behind is the easy remainder. It is recorded here only to explain the source file.

Between A and B: A credits the defense for 145 pairs on 71 images it never touched. B does not.
B is the honest one, and it is the denominator the summaries lead with.

## Relative reduction — and why there are three of them

Only gemini-3.1-flash-lite has both conditions, and 17 images appear in both its runs. That overlap
is the only true before/after in this folder: same model, same images, same 180 ground-truth pairs,
scored the same way.

| comparison | raw 6w | defended 6w | reduction 6w | reduction 3w |
|---|---|---|---|---|
| **paired** — 17 images, 180 pairs | 0.6667 | 0.1444 | **78.3%** | 77.9% |
| cross-set A — raw 200 imgs vs defended /1518 | 0.6827 | 0.0329 | 95.2% | 95.2% |
| cross-set B — raw 200 imgs vs defended /1373 | 0.6827 | 0.0364 | 94.7% | 94.7% |

The three differ by ~17 points, and the difference is entirely in what each compares against. The
two cross-set rows put a raw number measured on a 200-image sample (dominated by
`skip_all_relationship` images, which are the easy ones) against a defended number whose denominator
is the full corpus, most of which the attacker was never shown. The paired row holds the image set
fixed and lets only the annotation change, which is the thing the defense actually varies.

**Cite the paired figure.** 78.3% six-way / 77.9% three-way, n=17 images, is what these files
support without an accounting assumption doing the work. The cross-set figures are reproducible
from the same files — `summary.json` and `MANIFEST.json` record all three — but they are not
before/after comparisons, and the gap between 78% and 95% is the size of the assumption.

## Response shape

Both parts share one schema:

```json
{
  "img_id": "10087",
  "predictions":   {"1 2": 1},
  "ground_truth":  {"1 2": 1},
  "raw_response": "[P1-P2]: Both individuals are focused on the same task ... => friends",
  "timestamp": 1779942181.450225
}
```

`raw_response` is the reasoning: one `[Pi-Pj]: <argument> => <label>` line per pair.
Process responses add six bookkeeping fields — `missed_pairs`, `subjects_matched`,
`subjects_unmatched`, `num_preds_detected`, `num_preds_matched`, `num_gt_subjects` — which exist
only because that stage detects its own subjects.

**One trap in the defended responses.** Their `ground_truth` field is the *filtered* ground truth
from `def_img/dataset.json`, which keeps only pairs whose subjects both survived — it is empty for
9 of the 41 images. Scoring against that field reproduces the misleading 80-pair number. Every
figure in this folder rescores against the original `dataset_large.json` ground truth instead, which
is what `index.csv` reports per image.

## What was checked

Run against the source caches; all passed:

- md5 of every copied file equals its source — 897/897
- all 897 responses parse as JSON — 0 unparseable
- `process/` image ids equal the 454 in `dataset_large.json` exactly
- the `defense.py` replay reproduces `def_img/dataset.json` — 41/41 images, surviving-pair labels identical
- both defended runs cover exactly the 41 protected images — no more, no fewer
- recomputed accuracy equals the source `metrics.json` for all five runs (0.6827, 0.7311, 0.6250, 0.8125, 0.5323) at matching pair counts
- the four defense outcomes partition the 1518 ground-truth pairs exactly (396 + 824 + 145 + 153)

`MANIFEST.json` records the same checks in machine-readable form, along with the source path each
file was copied from.
