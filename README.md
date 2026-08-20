# AnyCloud Trainer

Run WMO-compatible LoRA training on a Lambda GPU through AnyCloud, with resumable checkpoints and
a portable PEFT adapter that survive service restarts.

This repository contains an authenticated, stateful training service. A CPU-side client keeps the
dataset and training loop, while the service performs tokenization, forward and backward passes,
optimizer updates, and checkpoint serialization on the remote GPU.

The first integration target is the existing `TrainerBackend` contract in Experiential Labs'
World Model Optimizer. The service is deliberately narrow:

- one exclusive LoRA session per GPU service;
- final-assistant cross-entropy supervision;
- one explicit optimizer update per request;
- resumable optimizer state uploaded to a caller-authorized object URL; and
- a portable PEFT adapter as the final sampling artifact.

## Run the validated image

Create an AnyCloud named secret called `anycloud-trainer` with one entry, `TRAINER_TOKEN`, then run:

```bash
anycloud service ghcr.io/anycloud-sh/anycloud-trainer@sha256:1f18acebd0918a74c7d0c6955e8f351a1276d84878fc2c819445397b26fb258f --id wmo-anycloud-trainer --credentials lambda --region us-east-1 --vm-type gpu_1x_a10 --secret anycloud-trainer --disk-size 160 --gpus all
```

The public GHCR package is connected to this repository. Release tag `v0.1.0` points to the same
validated OCI index; the digest above is the immutable form. AnyCloud assigns `PORT=8088` and
exposes the service at `https://wmo-anycloud-trainer.anycloud.sh`.

## Lambda evidence

The image was validated on three separate Lambda `gpu_1x_a10` services in `us-east-1` against
`Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

1. train one real WMO LoRA step and upload a 195,391,045-byte resumable checkpoint;
2. restore that checkpoint in a fresh GPU process, train again, and export a 57,526,030-byte PEFT
   adapter; and
3. load the adapter in a third fresh GPU process and render the WMO example again.

Both training losses and gradient norms were finite, CUDA execution was verified in every process,
and every service was terminated and provider-cleaned after its phase. The three cold-start
services cost an estimated $0.7554 in total at Lambda's reported $1.29/hour rate. See the
machine-readable
[`Lambda A10 validation receipt`](validation/lambda-a10.json) for image, source, model, API,
artifact, timing, and cost facts.

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
returned as model handles or written to WMO evidence.

The matching WMO [`AnyCloudTrainerBackend`](https://github.com/anycloud-sh/world-model-optimizer/tree/25af24b478e307e565d68afb9dacb53f11ac6469)
uses the existing injected `TrainerBackend` seam. The caller retains the authenticated HTTP client
and object-store authority; WMO persists only opaque `s3://` resource identities. The
[`validation controller`](validation/run_wmo_adapter.py) exercises initial training, restart from
optimizer state, and a fresh load of the exported PEFT adapter in separate GPU services.

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

This is an AnyCloud-powered training runtime, not a fork or redistribution of WMO. The thin WMO
client adapter lives on the `anycloud-trainer-backend` branch of the native GitHub fork. WMO's
upstream repository currently has no software license, so that fork is maintained only as a
proposed upstream contribution, pinned from upstream revision
`b7593f7e5a1da047012edde1981e3efec34f0d9c`.

Licensed under Apache-2.0.
