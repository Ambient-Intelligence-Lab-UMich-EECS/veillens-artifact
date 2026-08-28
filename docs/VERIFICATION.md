# Verification

The loop is *parameterised* by granularity: `experiment.mask_level` says which cue granularities to
remove. Verification is the other half of that contract — the check that the granularities the run
promised to remove are actually gone.

This matters because the two can drift. A run configured to strip street- and city-level cues but
checked against an unrelated distance is not verified at the granularity it targeted; it is only
measured. So there is **one knob**, `mask_level`, and everything downstream is derived from it.

```
mask_level: ["street level clue", "city level clue"]
        │
        ├─ what gets removed .......... street- and city-level cues
        ├─ coarsest protected level ... city level clue
        ├─ target rung ................ city
        └─ pass threshold ............. 10 km
```

There is no second threshold to keep in sync, which is why `experiment.val_dist` no longer exists.
A config that still sets it is rejected at load, with the migration in the error message.

## The granularity ladder

| Rung | Within | Targetable |
|---|---|---|
| block | 0.5 km | — |
| street | 1 km | ✅ `street level clue` |
| neighborhood | 5 km | — |
| city | 10 km | ✅ `city level clue` |
| regional | 100 km | ✅ `country level clue` |

Five rungs, three targets. The oracle schema has three clue levels, so `block` and `neighborhood`
are reporting-only: they appear in the threshold table `veillens-verify` prints, and they can never
be a target. The ladder lives in one place, `veillens/verify/granularity.py:LADDER_KM`, and a test
pins it.

## Two criteria

An image passes when both hold. They answer different questions and they disagree often, which is
itself worth reporting.

**Metric** — the attacker's estimate is at least `threshold_km` from the truth.

This is the criterion the reported numbers use. It needs the dataset label, so it is an offline
measurement and never something a deployed system could compute. It says nothing about granularity
beyond picking the radius.

**Residual** — the attacker's own report names no surviving cue at a protected granularity.

This needs no ground truth at all. It reads `critical_features[].properties` (geolocation and
social relationship) or `critical_features[].severity_level` (compositional) — labels the attacker
already emits under its schema. It is the criterion that is actually *about* granularity, and it is strictly the stricter
of the two: an image can be 400 km mislocated and still have the attacker pointing at a legible
street sign.

Both are three-valued. A criterion that could not be computed is `null`, and `null` never becomes a
pass — missing evidence is not evidence of protection. Likewise the residual check **fails closed**:
a cue whose granularity label is missing or unparseable counts as residual and is flagged. A
well-formed cue at an *out-of-scope* granularity does not: a country-level cue surviving city-level
protection was never targeted.

## Running it

```bash
veillens-verify runs/geolocation/my_run
```

Offline — no API key, no GPU, no weights, and no config file. A run records the granularity it
targeted in its own results (`feature_removed` is the formatted `mask_level`), so an archived run
carries the promise it should be checked against. `--mask-level` overrides when you want to check a
run against a different target than it was built for.

A live run also writes `evaluation_results/verification.json` itself, at the end of Phase 5, from
verdicts computed during evaluation — so it costs no extra API calls.

The residual criterion needs the attacker's raw responses, which `evaluation_results.json` does not
retain (it keeps only the location estimate). `veillens-verify` finds them in `eval_cache/` or
`evaluation/<model>/responses/`. If it finds none it reports the metric verdict and says the
residual one was **not computed** — it does not treat a missing response as a clean one. If it finds
several it refuses to guess and lists them for `--responses`.

### What a residual cue is, and is not

A residual cue is the attacker's own account of the evidence class it relied on. It is **not** proof
that the true location at that granularity is recoverable. An attacker can report a street-level cue
and still be 200 km wrong — it found street-scale evidence and drew the wrong conclusion from it.

So the residual criterion measures *whether granular evidence survived the edit*, which is what the
defense claims to remove, rather than *whether the adversary succeeded*, which is what the metric
criterion measures. That is exactly why both are reported and why their disagreement is printed: on
a 4-image smoke run, three images were ≥ 10 km off and all four still had street- and city-level
cues in the attacker's report.

## The attacker prompt has to permit an empty answer

The residual criterion counts what the attacker reports, so it is only meaningful if the attacker is
free to report nothing. `prompts/geolocation/attack_base_prompt.txt` previously ended:

> There should be at least one feature match to one level!!! There should be at least one street
> level clue!!!

Under city-level protection that makes the residual criterion **unpassable by construction** — every
response is required to contain a street-level clue, and street level is protected. Responses
produced under that instruction cannot be used to evaluate residual leakage, which is why
`veillens-verify` prints a warning when every image fails.

The prompt now asks for the clues the attacker actually relied on and states that an empty
`critical_features` list is a valid answer. This is a change to the measurement instrument: numbers
produced before it are not comparable to numbers produced after.

## Verifying a baseline

A prior-work defense targets no granularity — it perturbs the image and that is all — so its
results carry no `mask_level` to recover. Name the target you want it judged against and the same
two criteria apply, which is what makes the comparison in [BASELINES.md](BASELINES.md) a comparison
rather than two different measurements:

```bash
veillens-verify runs/geolocation/baselines/geoshield \
    --mask-level "street level clue,city level clue"
```

The attacker's readings of the defended images live in the method's `attacker_cache/`, which
`veillens-verify` finds on its own. (The shared `_attacker_cache/` of *unprotected* originals is a
different directory and is not picked up.)

## Compositional privacy

Compositional risk is a unitless CPRT score, not a distance, so it has no rung and no metric
criterion — `target_rung` is `None` and the loop has no distance-based early stop. Its verification
is the residual criterion alone: `mask_level: 2` passes when no cue at severity 1 or 2 survives,
which is exactly the paper's "suppressing both level 1 and 2 identifiers".

No score pass-line is invented here. The paper reports mean score and relative reduction for this
setting and defines no per-image failure threshold, so `veillens-verify` reports the distribution
beside the residual verdict and leaves it at that.

## Social relationship

`mask_level: ["intimate level cue", "non-intimate level cue"]` is the paper's target: *"We set the
social-relationship target to no relationship, suppressing evidence for both intimate and
non-intimate relationship inference."* Both criteria apply, and neither is a distance.

**Metric** — the attacker recovers none of the image's annotated pairs. The per-image error is the
share of pairs it got wrong, so the pass line is 1.0. `target_rung` is `None`, `threshold_km` is
`None`, and `metric_threshold` carries the 1.0 — `threshold_km` stays kilometres-only so its name
never has to be read as anything else.

An unanswered pair counts as not recovered. That is the conservative reading for the *metric*, but
it is generous to the defense in aggregate: an adversary that returns nothing at all would score as
perfect protection. So `veillens-score` reports relationship accuracy over two denominators — every
annotated pair, and answered pairs only — and prints how many pairs went unanswered. They diverge
exactly when the protected image stops the adversary committing, which is a real effect but a
different claim from getting the answer wrong.

**Residual** — the attacker cites no surviving cue tagged `intimate level cue` or
`non-intimate level cue`. The schema admits a third value, `no relationship`, which is deliberately
not protectable: it is the answer the defense is trying to push the adversary towards, so counting
a surviving no-relationship cue as a leak would make the criterion unpassable by construction. It
is recognised as a valid label and treated as out of scope, exactly like a country-level cue under
city-level protection.

Accuracy is reported six-way over PISC's own labels and three-way over the standard grouping
(`friends`/`family-members`/`couple` → intimate, `professional`/`commercial` → non-intimate,
`no-relationship` → no relation). Both are recomputed from the same responses rather than read from
separate runs; the grouping lives in one place, `veillens/metrics/relationship.py:THREE_WAY`.

## What changed, and what it invalidates

The stop rule moved from a hand-set `val_dist: 5.0` to the `mask_level`-derived threshold, which for
the shipped geolocation config is **10 km**. That is stricter: images that used to stop between 5 and
10 km now keep iterating. Combined with the attack-prompt fix, both the defense's behaviour and the
attacker's instrument have changed, so **every geolocation number produced before this change needs
re-running**. The archived runs under `results/` say so in their own READMEs.
