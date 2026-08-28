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
├── models/
│   ├── sam2/sam2.1_hiera_large.pt          857 MB
│   └── gdino/groundingdino_swint_ogc.pth   662 MB
└── hf_cache/                               diffusion weights, fetched on first use
```

## Datasets

| Dataset | Used by | Account needed | Where |
|---|---|---|---|
| **Cityscapes** | geolocation | yes | <https://www.cityscapes-dataset.com/> — you need `leftImg8bit_trainvaltest` **and** the vehicle metadata package, which carries the GPS |
| **VISPR** | compositional privacy | no | <https://tribhuvanesh.github.io/vpa/> — the images CPRT-Bench annotates |

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

## Verifying an install

```bash
veillens-protect --config configs/geolocation.yaml --dry-run
```

lists every missing file — and every unbuilt CUDA extension — with the command to fix it. Run it
before starting a long job: the GroundingDINO extension in particular fails at inference rather
than at import, so without this check a run gets all the way to a score table before the problem
shows.
