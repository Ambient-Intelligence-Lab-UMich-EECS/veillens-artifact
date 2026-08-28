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
| **Verify** | The oracle re-reads the edited image; if it still leaks, the loop runs again | `src/veillens/loop/` |

Because the reasoning step is explicit, you can see *why* something was flagged rather than getting
an opaquely blurred photo back.

Two privacy settings ship here: **geolocation** and **compositional privacy**.

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

```bash
pip install --no-build-isolation -e third_party/Grounded-SAM-2
pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
```

This needs `nvcc` and `CUDA_HOME` set. `--dry-run` (below) checks the extension is built, so run
it before anything long.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you intend to run the loop.

## Try it with no API key and no GPU

`veillens-score` reads the JSON a run already wrote, so the bundled examples work immediately:

```bash
veillens-score examples/results/geolocation/evaluation_results.json
veillens-score examples/results/compositional/run_evaluation_results.json --setting compositional
```

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
  privacy_class: geolocation                              # geolocation | compositional
  mask_level: ["street level clue", "city level clue"]    # which cues to remove
  max_iter: 3                                             # reason→edit→verify rounds
  val_dist: 5.0                                           # stop once the oracle is this far off (km)
```

---

## Repository layout

```
src/veillens/     the package — reason/ ground/ edit/ loop/ privacy/ …
configs/          one file per privacy setting
prompts/          the prompts and JSON schemas the oracle is given, verbatim
manifests/        image identifiers and ground-truth labels for the evaluated sets
examples/         per-example records from finished runs, for veillens-score
results/          every VLM response behind a finished experiment, one file per response
tests/            offline test suite: no key, no GPU, no data
docs/             usage, architecture, and where the data comes from
third_party/      the Grounded-SAM-2 submodule — needs building, see Install
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
| [docs/DATA.md](docs/DATA.md) | Datasets and weights: source, licence, target directory, size |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Third-party components and their licences |

## Licence

MIT for the first-party code in this repository — see [LICENSE](LICENSE). The submodule, the model
weights and the datasets keep their own terms; `THIRD_PARTY.md` spells out which.
