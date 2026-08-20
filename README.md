# AnyCloud Trainer

Run WMO-compatible LoRA training on a GPU in a cloud account selected through AnyCloud.

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

## Run

The documented immutable image digest will be added after the candidate passes the Lambda GPU
validation. The launch shape is:

```bash
anycloud service \
  ghcr.io/anycloud-sh/anycloud-trainer@sha256:{{validated_digest}} \
  --id wmo-anycloud-trainer \
  --credentials lambda \
  --gpu-type a6000 \
  --secret anycloud-trainer \
  --disk-size 160
```

The named secret contains one entry, `TRAINER_TOKEN`. AnyCloud assigns `PORT=8088` and exposes the
service at `https://wmo-anycloud-trainer.anycloud.sh`.

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

The container uses PyTorch 2.8 with CUDA 12.8. The initial validation model is
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
