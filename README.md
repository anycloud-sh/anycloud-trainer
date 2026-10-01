# AnyCloud Trainer

Run Experiential-compatible LoRA training on a Lambda GPU through AnyCloud, with resumable checkpoints and
a portable PEFT adapter that survive service restarts.

This repository contains an authenticated, stateful training service. A CPU-side client keeps the
dataset and training loop, while the service performs tokenization, forward and backward passes,
optimizer updates, and checkpoint serialization on the remote GPU.

The first integration target is the `TrainerBackend` contract in Experiential Labs' open source
[`experiential`](https://github.com/experientiallabs/experiential) project. The service is deliberately narrow:

- one exclusive LoRA session per GPU service;
- final-assistant cross-entropy supervision;
- one explicit optimizer update per request;
- resumable optimizer state uploaded to a caller-authorized object URL; and
- a portable PEFT adapter as the final sampling artifact.

## Run the validated image

Create an AnyCloud named secret called `anycloud-trainer` with one entry, `TRAINER_TOKEN`, then run:

```bash
anycloud service ghcr.io/anycloud-sh/anycloud-trainer@sha256:1f18acebd0918a74c7d0c6955e8f351a1276d84878fc2c819445397b26fb258f --id exp-anycloud-trainer --credentials lambda --vm-type gpu_1x_a10 --secret anycloud-trainer --disk-size 160 --gpus all
```

The public GHCR package is connected to this repository. Release tag `v0.1.0` points to the same
validated OCI index; the digest above is the immutable form. AnyCloud assigns `PORT=8088` and
exposes the service at `https://exp-anycloud-trainer.anycloud.sh`.

## Current-code validation

On 2026-10-01, the current Experiential `TrainerBackend` adapter completed the three-phase GPU
validation against the same immutable trainer image. The first Lambda A10 service trained and
saved a resumable checkpoint. After Lambda capacity prevented a second service from starting, an
AWS A10G service restored the checkpoint, trained another step, and exported a PEFT adapter. A
fresh AWS A10G service loaded that adapter and rendered the same example. The
[`current-code validation receipt`](validation/experiential-2026-10-01.json) records the source
revisions, results, capacity failures, and cleanup state. All three successful services were
provider-cleaned; estimated total GPU cost was $0.4474. This validates the injected Python backend,
not the Tinker-only `exp optimize model` CLI path.

## Earlier Lambda evidence

The image was validated on three separate Lambda `gpu_1x_a10` services in `us-east-1` against
`Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

1. train one real LoRA step against the former WMO source and upload a 195,391,045-byte resumable checkpoint;
2. restore that checkpoint in a fresh GPU process, train again, and export a 57,526,030-byte PEFT
   adapter; and
3. load the adapter in a third fresh GPU process and render the example again.

Both training losses and gradient norms were finite, CUDA execution was verified in every process,
and every service was terminated and provider-cleaned after its phase. The three cold-start
services cost an estimated $0.7554 in total at Lambda's reported $1.29/hour rate. See the
machine-readable
[`August Lambda A10 validation receipt`](validation/lambda-a10.json) for image, source, model, API,
artifact, timing, and cost facts. That receipt predates the rename to `experiential`.

## HTTP contract

Every mutation request uses `Authorization: Bearer $TRAINER_TOKEN`.

```text
POST /v1/sessions
POST /v1/sessions/{session_id}/datums:render
POST /v1/sessions/{session_id}/batches:train
POST /v1/sessions/{session_id}/artifacts:save
```

`GET /healthz` is public and reports only liveness, CUDA availability, and whether a session is
active. Artifact upload and restore URLs are short lived. They are used in memory and never
returned as model handles or written to Experiential's SFT evidence.

The matching [`AnyCloudTrainerBackend`](https://github.com/anycloud-sh/world-model-optimizer/tree/anycloud-trainer-backend-v2)
uses the existing injected `TrainerBackend` seam. The caller retains the authenticated HTTP client
and object-store authority; Experiential persists only opaque `s3://` resource identities. The
[`validation controller`](validation/run_experiential_adapter.py) exercises initial training, restart from
optimizer state, and a fresh load of the exported PEFT adapter in separate GPU services.

The current `exp optimize model` command constructs Tinker's backend directly. The AnyCloud adapter
is available through the Python `TrainerBackend` seam and has not been wired into that CLI command.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
```

The container uses PyTorch 2.8 with CUDA 12.8. The validated model is
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using Transformers 5.5.3 to
match the current Tinker publishing path.

## Scope

This is an AnyCloud-powered training runtime, separate from Experiential. The thin client adapter
lives on the `anycloud-trainer-backend-v2` branch of the native GitHub fork. Experiential is now
licensed under Apache-2.0. The adapter remains a proposed integration, with no upstream PR opened.

Licensed under Apache-2.0.
