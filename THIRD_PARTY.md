# Third-party components

First-party code in this repository is MIT (see `LICENSE`). Everything below has its own terms.

## Vendored as a submodule

| Component | Pin | Upstream | Licence |
|---|---|---|---|
| **Grounded-SAM-2** | `2c0178f` | <https://github.com/IDEA-Research/Grounded-SAM-2> | Apache-2.0. Bundles SAM 2 (Meta, Apache-2.0) and GroundingDINO (IDEA-Research, Apache-2.0), each with its own licence file |
| **GeoShield** | `5001a82` | <https://github.com/thinwayliu/Geoshield> | MIT. Used only by the `geoshield` baseline, and only for its CLIP feature extractors and ensemble loss — see `docs/BASELINES.md` for exactly which parts are upstream's |

Pinned to an exact commit rather than a branch, so upstream changes cannot silently alter
behaviour.

## Model weights (downloaded, not redistributed)

| Model | Licence |
|---|---|
| SAM 2.1 Hiera-Large | Apache-2.0 (Meta) |
| GroundingDINO Swin-T OGC | Apache-2.0 (IDEA-Research) |
| SDXL Inpainting 1.0 | CreativeML Open RAIL++-M — carries use restrictions; read them |
| ControlNet depth SDXL | OpenRAIL |
| MiDaS (DPT-Hybrid) | MIT |

The SDXL OpenRAIL++-M licence in particular restricts certain uses.

Baselines only:

| Model | Used by | Licence |
|---|---|---|
| EAST text detector (frozen graph) | `ocr-blur` | Apache-2.0 (the detector is Zhou et al. 2017; the frozen graph is the mirror OpenCV's own sample points at) |
| Places365 ResNet-18 / ResNet-50 | `mm-pgd` | CC BY (MIT CSAIL) — <http://places2.csail.mit.edu/> |
| CLIP ViT-B/16, ViT-B/32 | `geoshield` | MIT (OpenAI) |
| CLIP ViT-G/14 (LAION-2B) | `geoshield` | MIT (LAION) |

## Datasets (downloaded, not redistributed)

| Dataset | Licence / terms |
|---|---|
| **Cityscapes** | Account required; redistribution prohibited. <https://www.cityscapes-dataset.com/license/> |
| **VISPR** | CC BY-NC 4.0 — non-commercial only |
| **CPRT-Bench** | Annotations only; see the dataset card. The 500-row subset used here is in `manifests/` |

`manifests/` contains image identifiers and ground-truth labels only — no pixels — which is why
it can be redistributed when the images themselves cannot.
