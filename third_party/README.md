# Third-party checkouts

## Grounded-SAM-2

Stages II and III need open-set detection (GroundingDINO) and promptable segmentation (SAM 2).
Upstream ships no installable package, so it is vendored as a submodule pinned to an exact commit:

```bash
git submodule update --init --recursive third_party/Grounded-SAM-2
```

If you already have a checkout elsewhere, point at it instead and skip the submodule:

```bash
export VEILLENS_GSAM2=/path/to/Grounded-SAM-2
```

`veillens-protect --dry-run` reports which of the two you have, and what is missing.

Licences are in `THIRD_PARTY.md`. The model checkpoints are downloaded separately — see
`docs/DATA.md`.
