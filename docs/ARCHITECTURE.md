# Architecture

The four stages of the paper map onto the package like this:

| Stage | Module | Entry point |
|---|---|---|
| I — reason | `reason/oracle.py` | `PrivacyOracle.get_geolocation_prediction` |
| II — ground | `ground/segmenter.py` | `GroundedSAM2Segmenter.segment` |
| III — edit | `edit/{pipeline,inpaint,control}.py` | `ControlNetInpainter.inpaint_batch` |
| IV — verify | `verify/granularity.py`, `loop/{driver,api_workers}.py` | `verify.verdict`, the gate and final-eval tasks |

Stage IV is split, and the split is deliberate. *Re-reading* the edited image is the same oracle
from stage I, so it stays in the loop that schedules it — the gate and the final-eval task. *Deciding
whether the result counts as protected* is `verify/`, which is pure functions over a run's own JSON:
no model, no network, no GPU. That is what lets `veillens-verify` re-derive the verdict for a
finished run offline, and what keeps the pass criterion in one place instead of spread across a
config value, a gate comparison and a reporting threshold. See [VERIFICATION.md](VERIFICATION.md).

## Why there are three queues

The three stages have incompatible resource profiles. Stage I is network-bound and wants many
requests in flight. Stage II is GPU-bound and single-threaded, because SAM 2 holds one model on one
device. Stage III is GPU-bound and wants to batch, because a diffusion forward pass amortises badly
over one image. Running them in one sequential loop would leave the GPU idle during every API call
and the network idle during every diffusion step.

So `loop/queues.py` defines three:

```
api_queue   asyncio.Queue    stage I + IV  — VLM calls, many concurrent
seg_queue   queue.Queue      stage II      — one GPU thread, strictly serial
sd_queue    GlobalBatchQueueManager        — stage III, batched across images
```

An image is a state machine (`loop/state.py`) that moves between them. `loop/driver.py` owns the
event loop, drains `sd_queue` when a batch fills, and holds a GPU lock so segmentation and
diffusion never contend for the device.

`sd_queue` batches *across* images rather than within one, because a single image rarely has enough
cues of the same size to fill a batch. `loop/batching.py` assigns each cue a size class from its
bounding box — `small`/`median`/`large` get a fixed square crop, `full` gets the whole frame — and
each class has its own queue, since a diffusion batch requires identical dimensions.

## One iteration

```
      manifest
         │
         ▼
   ┌───────────┐   R1_EVAL      the oracle lists cues: what is inferable, the object to look
   │ api_queue │ ─────────────▶ for, the concept to replace it with, and why it matters
   └───────────┘
         │ seg task
         ▼
   ┌───────────┐   Grounding DINO locates each cue's object description; SAM 2 turns the
   │ seg_queue │   surviving boxes into masks; MiDaS depth is computed once and cached
   └───────────┘
         │ one batch item per cue
         ▼
   ┌───────────┐   SDXL inpaints each masked region under depth-ControlNet conditioning,
   │ sd_queue  │   batched by size class; results are composited back into the frame
   └───────────┘
         │ composited image
         ▼
   ┌───────────┐   GATE_EVAL     the oracle re-reads the edited image. Still leaking, and
   │ api_queue │ ──────────────▶ budget left? Round 2 feeds new cues back into seg_queue.
   └───────────┘                 Otherwise write final.png and run FINAL_EVAL.
```

Round 2 has two modes. With `fine_grained: false` the whole image is re-read each iteration. With
`fine_grained: true` the frame is divided into overlapping crops by
`loop/seg_worker.py:_calculate_sliding_window_regions`, each is upscaled and re-read separately, and
the cues found in each are grounded in that crop's coordinate frame.

## What makes it generic

`privacy/spec.py` holds a `PrivacyClassSpec`: the prompt and schema paths, which cues a given mask
level admits, how round-2 context is phrased, and what "error" means for that setting. The loop
never branches on a class name — it asks the spec. `privacy/registry.py` maps names to specs and
keeps the legacy aliases (`personal_info`, `pi`, `geo`) resolving, because cached result records
were written under the older spellings.

Adding a privacy class means writing one spec and registering it. Nothing in `loop/` changes.

## Grounding without a detector

Stage II has two entrances, and which one a clue takes is a property of the clue rather than of the
privacy class. The usual one is `segment()`: Grounding DINO proposes boxes for the clue's language
description and SAM 2 refines them. The other is `segment_boxes()`, which takes boxes already in
hand and goes straight to SAM 2.

The social-relationship setting uses the second, because PISC ships person boxes and subject ids and
the paper's intervention policy is to use them: *"we use dataset-provided person boxes and IDs to
isolate relationship reasoning from person discovery."* Its spec implements `feature_boxes`, which
maps a nominated `subject_id` to that subject's box and narrows it to a head band; the seg worker
asks every spec for boxes first and falls through to detection when it gets `None`. With
`detection.use_grounding_dino: false` the detector is never loaded, so that setting needs neither
the SwinT weights nor the compiled deformable-attention kernel.

Two other seams exist for the same reason. `format_item_context` appends per-image text to every
prompt — the social-relationship prompts have to name which numbered subjects form an annotated
pair, and only the manifest knows that. `eval_original` asks Phase 5 to read the *unedited* image
with the attacker rather than reusing the defense oracle's round-1 response, which a setting needs
when its headline number is one adversary's accuracy before versus after. Both default to the
existing behaviour, so the two shipped settings are unaffected.

## The VLM layer

`vlm/base.py` defines the client interface and image encoding; `vlm/openai_client.py` implements it
against the OpenAI Responses API with strict structured output; `vlm/registry.py` builds a client
from a config section. `reason/oracle.py` holds two clients — the defense oracle and the attacker —
and routes each call by *role*, never by schema, so that a setting which happens to reuse one schema
for both cannot silently bill a measurement to the wrong model.
