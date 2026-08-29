# VeilLens

**Find out what a vision-language model can infer from your photo — and remove it.**

Modern VLMs read far more from an image than the obvious identifiers. Not just a face or a licence
plate, but the street you were standing on, or how much of your private life a scene gives away.
They do it by *composing* weak cues: a bus livery plus a kerb style plus a shopfront font.

VeilLens takes an image and a privacy goal and runs a four-stage loop:

| Stage | What happens | Code |
|---|---|---|
| **Reason** | A VLM oracle says what is inferable, and which visual evidence supports it | `src/veillens/reason/` |
| **Ground** | Each cue, described in words, becomes a pixel mask (Grounding DINO → SAM 2) | `src/veillens/ground/` |
| **Edit** | The selected cues are inpainted away (SDXL + depth ControlNet), preserving the rest | `src/veillens/edit/` |
| **Verify** | The oracle re-reads the edited image; if it still leaks at the granularity you asked for, the loop runs again | `src/veillens/verify/`, `src/veillens/loop/` |

Because the reasoning step is explicit, you can see *why* something was flagged rather than getting
an opaquely blurred photo back.

Three privacy settings ship here: **geolocation**, **compositional privacy**, and
**social relationship**. The third differs in one structural way: PISC supplies person boxes and
subject ids, so it skips open-set detection and anonymizes the faces of the subjects that carry a
relationship — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Install

```bash
git clone --recurse-submodules <repo-url> veillens
cd veillens
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: config validation + scoring, no GPU needed
pip install -e '.[all]'     # everything: grounding, editing
```

To run the loop you also need SAM 2 and GroundingDINO, which are not on PyPI and come from the
submodule. **Both build CUDA extensions, and GroundingDINO's is not optional:** without it every
detection fails at inference time, so the loop grounds nothing, edits nothing, and still finishes.

```bashX
pip install --no-build-isolation -e third_party/Grounded-SAM-2
pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
```

This needs `nvcc` and `CUDA_HOME` set. `--dry-run` (below) checks the extension is built, so run
it before anything long.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you intend to run the loop.

## Watch it run

[`demo.mp4`](demo.mp4) walks a recorded geolocation run through the loop one stage at a time —
the oracle naming what it can infer and from which cues, each cue being grounded to a mask, the
choice of what to remove, the edit, and the verification gate deciding whether to iterate. Two and
a half minutes, and it needs nothing installed.

## Try it with no API key and no GPU

`examples/results/` carries two images per privacy setting, copied out of real runs: the aggregate,
the attacker's raw reply, and the before/after pair. Both offline commands take the directory:

```bash
veillens-score  examples/results/geolocation        # also: compositional, social_relation
veillens-verify examples/results/geolocation
```

`veillens-score` reports the distribution — how far off the attacker was. `veillens-verify` reports
the verdict — whether the cue granularities the run promised to remove are actually gone, by two
independent criteria. Both read the run's own JSON, so neither needs a key or a GPU.

Two images per setting show the format and the edit; they are not evidence of effect size. The
reported numbers come from the full runs in [`results/`](results/), and
[`examples/results/README.md`](examples/results/README.md) says what each sample does and does not
demonstrate — including the social-relationship pair, where the attacker is *not* fooled.

## Check an install before spending anything

```bash
veillens-protect --config configs/geolocation.yaml --dry-run
```

This validates the config, resolves every prompt, schema, checkpoint and dataset path, and reports
exactly what is missing — without one API call or a second of GPU time. Run it first.

## Protect a set of images

```bash
veillens-protect --config configs/geolocation.yaml
```

Two configs ship, one per privacy setting, and each is short enough to read end to end. The values
you are most likely to change:

```yaml
experiment:
  privacy_class: geolocation                # geolocation | compositional | social_relation
  mask_level: ["street level clue", "city level clue"]    # which cues to remove
  max_iter: 3                                             # reason→edit→verify rounds
```

`mask_level` is the only protection knob, and verification is derived from it: removing street- and
city-level cues is city-level protection, so the loop iterates until the attacker is at least 10 km
wrong and every image is checked against 10 km. There is no second threshold to keep in sync.

## Compare against prior work

Five prior-work defenses ship alongside the loop, and run on the same images scored by the same
attacker — which is the only way the comparison means anything:

```bash
pip install -e '.[baselines]'
python scripts/fetch_baseline_weights.py --method all
veillens-baseline --method all --config configs/geolocation.yaml --dry-run
veillens-baseline --method rand-noise --config configs/geolocation.yaml
veillens-score runs/geolocation/baselines/rand-noise
```

`rand-noise`, `ocr-blur`, `mm-pgd`, `geoshield` and `gpt-image`. Only `gpt-image` costs money;
`rand-noise` needs no weights and no GPU. See [docs/BASELINES.md](docs/BASELINES.md).

---

## Repository layout

```
src/veillens/     the package — reason/ ground/ edit/ loop/ privacy/ …
configs/          one file per privacy setting
prompts/          the prompts and JSON schemas the oracle is given, verbatim
manifests/        image identifiers and ground-truth labels for the evaluated sets
examples/         two worked images per privacy setting, for veillens-score and -verify
results/          every VLM response behind a finished experiment, one file per response
scripts/          one-off setup helpers (baseline weight download)
docs/             usage, architecture, baselines, and where the data comes from
third_party/      the Grounded-SAM-2 and GeoShield submodules — see Install
demo.mp4          screen recording of a full run
```

## Where data goes

Nothing large is committed. Point one variable at a disk with room and everything lands under it:

```bash
export VEILLENS_DATA=/path/to/veillens-data
```

`docs/DATA.md` lists every dataset and checkpoint, its source, its licence, and its target
directory.

## Documentation

| | |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Every entry point, every config key, reading the output tree |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the loop and its worker queues fit together |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | The granularity ladder, the two pass criteria, and what `mask_level` derives |
| [docs/DATA.md](docs/DATA.md) | Datasets and weights: source, licence, target directory, size |
| [docs/BASELINES.md](docs/BASELINES.md) | The five prior-work defenses: install, run, and what is faithful to upstream |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Third-party components and their licences |

## Licence

MIT for the first-party code in this repository — see [LICENSE](LICENSE). The submodule, the model
weights and the datasets keep their own terms; `THIRD_PARTY.md` spells out which.
