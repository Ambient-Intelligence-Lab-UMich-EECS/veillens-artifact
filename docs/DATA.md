# Data and model weights

Nothing large is committed. This page says where each piece comes from and which directory it
belongs in.

Point one variable at a disk with room, and everything lands under it:

```bash
export VEILLENS_DATA=/path/to/veillens-data
export HF_HOME=$VEILLENS_DATA/hf_cache      # keep HuggingFace downloads there too
```

It defaults to `<repo>/data`, which is gitignored — fine for a trial, too small for the full sets.

## The tree you are aiming for

```
$VEILLENS_DATA/
├── cityscapes/
│   ├── leftImg8bit/train/<city>/*.png      street scenes
│   └── vehicle/train/<city>/*.json         per-frame GPS — the geolocation ground truth
├── vispr/
│   └── test2017/2017_*.jpg                 compositional-privacy image pool  (17 GB)
├── pisc/
│   ├── image/%05d.jpg                      people-in-social-context photos   (7.3 GB)
│   ├── relationship.json                   pairwise relationship labels
│   ├── annotation_image_info.json          person boxes, one list per image
│   └── relationship_split/relation_*idx.json
├── pisc_annotated/<image_id>.jpg           written by the manifest builder, not downloaded
├── models/
│   ├── sam2/sam2.1_hiera_large.pt          857 MB
│   ├── gdino/groundingdino_swint_ogc.pth   662 MB
│   ├── east/frozen_east_text_detection.pb   92 MB   baselines only
│   └── places365/resnet{18,50}_places365.pth.tar    baselines only
└── hf_cache/                               diffusion and CLIP weights, fetched on first use
```

## Datasets

| Dataset | Used by | Account needed | Where |
|---|---|---|---|
| **Cityscapes** | geolocation | yes | <https://www.cityscapes-dataset.com/> — you need `leftImg8bit_trainvaltest` **and** the vehicle metadata package, which carries the GPS |
| **VISPR** | compositional privacy | no | <https://tribhuvanesh.github.io/vpa/> — the images CPRT-Bench annotates |
| **PISC** | social relationship | no | <https://zenodo.org/record/1059155> — People in Social Context; you need the images plus `relationship.json`, `annotation_image_info.json` and `relationship_split/` |

Both restrict redistribution, which is why neither is vendored here. `manifests/` carries the
image identifiers and ground-truth labels only, so the exact evaluated sets are pinned even though
the pixels are not shipped.

### Geolocation: building the manifest

```bash
python -m veillens.datasets.cityscapes \
    --cityscapes-root $VEILLENS_DATA/cityscapes \
    --output $VEILLENS_DATA/cityscapes/dataset.json \
    --images-per-city 25
```

This writes a schema-v2 manifest: `dataset` + `relpath` per item, resolved against
`$VEILLENS_DATA` at load time, so it stays portable between machines.

### Compositional privacy: VISPR images, CPRT-Bench labels

`manifests/compositional_500.json` is the 500-row subset of
[CPRT-Bench](https://huggingface.co/datasets/timtsapras23/CPRT-Bench)'s test split, with its
`score` column stored as `ground_truth_score`. The annotations are therefore already in this
repository; you do not need to download CPRT-Bench.

What you do need is the images, and CPRT-Bench does not ship them — it annotates **VISPR**
(Orekondy et al., *Towards a Visual Privacy Advisor*, ICCV 2017). No account, no form:

```bash
mkdir -p $VEILLENS_DATA/vispr && cd $VEILLENS_DATA/vispr
curl -LO https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz   # 17.4 GB
tar -xzf test2017.tar.gz          # unpacks to test2017/2017_########.jpg
```

The archive names `train2017` / `val2017` / `test2017` are a naming collision with COCO and nothing
more. CPRT-Bench's **test** split comes from VISPR's **test2017**. Licence is CC BY-NC 4.0 —
non-commercial, stricter than anything else in this tree.

Only 500 of the ~8,000 test images are referenced, so you can filter during a streaming extract and
never store the archive:

```bash
python -c "import json;[print('test2017/'+i['relpath']) for i in json.load(open('manifests/compositional_500.json'))['items']]" > /tmp/needed.txt
curl -sL https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz \
  | tar -xz -C $VEILLENS_DATA/vispr -T /tmp/needed.txt
```

That still transfers 17.4 GB, so it is a disk optimisation, not a bandwidth one.

### Social relationship: PISC boxes, labels, and the box overlay

```bash
python -m veillens.datasets.pisc \
    --pisc-root $VEILLENS_DATA/pisc \
    --annotated-dir $VEILLENS_DATA/pisc_annotated \
    --num-images 500 --seed 0 \
    --output manifests/social_relation_500.json
```

This does two jobs. It writes a schema-v2 manifest carrying each image's person boxes and its
annotated pair labels, and it renders `pisc_annotated/<image_id>.jpg` — the same photo with one
coloured, numbered rectangle per subject. The overlay is not decoration: the attack is *pairwise*,
so both the defense oracle and the adversary are asked about "P1 and P2", and without the drawn
boxes neither knows who that is.

Three things about PISC's own layout are easy to trip over:

* The image directory is **`image/`**, singular. Scripts elsewhere that say `images/` are written
  against a different unpacking of the same archive.
* **Person ids are positions.** There is no id field on a box; subject `k` is `bbox[k-1]`.
* **Ids are spelled two ways** — annotation keys are unpadded (`"599"`), image files are
  zero-padded (`00599.jpg`) — and `annotation_image_info.json` stores its id as an int while every
  other file uses a string.

Some distributions ship the images as split tarball chunks (`images-00`…`images-03`) rather than an
unpacked tree; reassemble with `cat images-0* | tar xzf -`. **Check that first**, because a partly
unpacked archive does not fail loudly: the builder simply skips the images it cannot find and
selects from what remains, which is a biased sample of the split rather than a random one. It
reports a `missing_image` count for exactly this reason, and records it in the manifest's
`provenance` block along with the split, the seed and the label histogram — so a manifest built
against an incomplete tree says so, forever.

## Model weights

| Weight | Size | Source |
|---|---|---|
| `sam2.1_hiera_large.pt` | 857 MB | Meta SAM 2 release, linked from the Grounded-SAM-2 README |
| `groundingdino_swint_ogc.pth` | 662 MB | IDEA-Research GroundingDINO release |

Fetched automatically from HuggingFace on first use, into `$HF_HOME`:

```
diffusers/stable-diffusion-xl-1.0-inpainting-0.1     ~6.5 GB
diffusers/controlnet-depth-sdxl-1.0-small            ~611 MB
```

MiDaS DPT-Hybrid comes from `torch.hub` on first use. Budget roughly **8 GB** for weights before
any dataset.

### Baselines

Only needed if you run `veillens-baseline`. One command fetches both, verifies them against a
recorded SHA-256, and puts them where `configs/baselines.yaml` expects:

```bash
python scripts/fetch_baseline_weights.py --method all      # 230 MB
python scripts/fetch_baseline_weights.py --check           # verify, download nothing
```

| Weight | Size | Used by | Source |
|---|---|---|---|
| `frozen_east_text_detection.pb` | 92 MB | `ocr-blur` | the frozen EAST graph OpenCV's text-detection sample points at |
| `resnet18_places365.pth.tar` | 43 MB | `mm-pgd` | <http://places2.csail.mit.edu/models_places365/> |
| `resnet50_places365.pth.tar` | 93 MB | `mm-pgd` | same |

`geoshield` needs no download here: its three CLIP surrogates come from HuggingFace on first use
(about **6 GB** into `$HF_HOME`, most of it LAION ViT-G/14), and its code comes from the
`third_party/GeoShield` submodule. `gpt-image` needs only `OPENAI_API_KEY`. See
[BASELINES.md](BASELINES.md).

## Verifying an install

```bash
veillens-protect --config configs/geolocation.yaml --dry-run
```

lists every missing file — and every unbuilt CUDA extension — with the command to fix it. Run it
before starting a long job: the GroundingDINO extension in particular fails at inference rather
than at import, so without this check a run gets all the way to a score table before the problem
shows.
