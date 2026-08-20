"""Drive bounded WMO adapter validation against successive AnyCloud GPU services."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

import boto3
import httpx
from pydantic import SecretStr
from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import AssistantAction
from wmo.optimize.model.sft.anycloud import (
    AnyCloudArtifactUpload,
    AnyCloudTrainerBackend,
    AnyCloudTrainerSession,
)
from wmo.optimize.model.sft.contracts import SFTExample, SFTMessage, TraceExampleSource
from wmo.optimize.model.sft.training import TinkerSFTSpec

MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
WMO_UPSTREAM_REVISION = "b7593f7e5a1da047012edde1981e3efec34f0d9c"
WMO_ADAPTER_REVISION = "25af24b478e307e565d68afb9dacb53f11ac6469"
_DIGEST = "a" * 64


@dataclass(frozen=True)
class _ObjectRecord:
    """Non-secret persisted object facts retained in validation evidence."""

    resource_id: str
    bucket: str
    key: str
    size_bytes: int
    etag: str
    version_id: str | None
    encryption: str | None


class _S3ArtifactStore:
    """Issue short-lived S3 transfers while retaining only opaque object identities."""

    def __init__(self, *, bucket: str, prefix: str, region: str) -> None:
        """Bind one caller-owned private validation prefix."""
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client("s3", region_name=region)

    def begin_upload(
        self, *, kind: Literal["state", "sampling"], name: str
    ) -> AnyCloudArtifactUpload:
        """Authorize one immutable object upload without exposing AWS credentials."""
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("artifact names must be one safe path component")
        key = f"{self._prefix}/{kind}/{name}"
        upload_url = self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ContentType": "application/octet-stream",
            },
            ExpiresIn=3600,
        )
        return AnyCloudArtifactUpload(
            resource_id=f"s3://{self._bucket}/{key}",
            upload_url=SecretStr(upload_url),
        )

    def download_url(self, resource_id: str) -> SecretStr:
        """Authorize one exact S3 resource identity for a bounded download."""
        bucket, key = _parse_s3_resource(resource_id)
        if bucket != self._bucket or not key.startswith(f"{self._prefix}/"):
            raise ValueError("resource identity is outside the validation prefix")
        return SecretStr(
            self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=3600,
            )
        )

    def inspect(self, resource_id: str) -> _ObjectRecord:
        """Return non-secret object metadata after a completed direct upload."""
        bucket, key = _parse_s3_resource(resource_id)
        response = self._client.head_object(Bucket=bucket, Key=key)
        return _ObjectRecord(
            resource_id=resource_id,
            bucket=bucket,
            key=key,
            size_bytes=int(response["ContentLength"]),
            etag=str(response["ETag"]).strip('"'),
            version_id=response.get("VersionId"),
            encryption=response.get("ServerSideEncryption"),
        )


class _ResponseJournal:
    """Retain safe response facts without request bodies or signed URLs."""

    def __init__(self) -> None:
        """Initialize an empty ordered journal."""
        self.records: list[dict[str, object]] = []

    def record(self, response: httpx.Response) -> None:
        """Read one response and retain its validated public JSON body."""
        response.read()
        body: object
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
        self.records.append(
            {
                "path": response.request.url.path,
                "status_code": response.status_code,
                "body": body,
            }
        )


def _parse_s3_resource(resource_id: str) -> tuple[str, str]:
    """Parse one opaque S3 identity without accepting URL authorization fields."""
    parsed = urlsplit(resource_id)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("resource identity must be an s3://bucket/key value")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("resource identity must not contain authorization")
    return parsed.netloc, parsed.path.lstrip("/")


def _spec() -> TinkerSFTSpec:
    """Return the exact bounded validation settings used in all three phases."""
    return TinkerSFTSpec(
        base_model=MODEL,
        lora_rank=8,
        learning_rate=0.00001,
        batch_size=1,
        epochs=1,
        checkpoint_every_steps=1,
        maximum_steps=2,
        maximum_datum_tokens=128,
    )


def _example() -> SFTExample:
    """Return one tiny accepted WMO example with an observable assistant target."""
    return SFTExample(
        example_id="anycloud-validation-example",
        leakage_group_id="anycloud-validation-lineage",
        task="Answer the arithmetic question with only the number.",
        history=(SFTMessage(role="user", content="What is two plus two?"),),
        target=AssistantAction(content="4"),
        source=TraceExampleSource(
            trace_id="anycloud-validation-trace",
            acceptance_evidence=ArtifactInput(
                artifact_id="anycloud-validation-acceptance",
                sha256=_DIGEST,
            ),
        ),
        source_step_index=0,
    )


def _client(service_url: str, token: str, journal: _ResponseJournal) -> httpx.Client:
    """Create one authenticated caller-owned service client with a bounded timeout."""
    return httpx.Client(
        base_url=service_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(1800, connect=60),
        event_hooks={"response": [journal.record]},
    )


def _backend(client: httpx.Client, store: _S3ArtifactStore) -> AnyCloudTrainerBackend:
    """Compose the exact WMO backend under validation."""
    return AnyCloudTrainerBackend(
        client,
        store,
        price_per_hour_usd=1.09,
        maximum_step_seconds=900,
        model_revision=MODEL_REVISION,
    )


def _run_training_phase(
    *,
    phase: Literal["initial", "resume"],
    service_url: str,
    token: str,
    store: _S3ArtifactStore,
    resume_resource_id: str | None,
) -> dict[str, object]:
    """Run one optimizer step and save state or a final sampling adapter."""
    journal = _ResponseJournal()
    started = time.monotonic()
    with _client(service_url, token, journal) as client:
        health_before = client.get("/healthz").json()
        session = _backend(client, store).open(_spec(), resume_resource_id)
        (datum,) = session.render_examples((_example(),))
        result = session.train_batch((datum,), learning_rate=_spec().learning_rate)
        if phase == "initial":
            resource_id = session.save_state("step-000001.pt")
        else:
            resource_id = session.save_sampling_handle("final-adapter.tar.gz")
        health_after = client.get("/healthz").json()
    return {
        "phase": phase,
        "started_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "service_url": service_url,
        "health_before": health_before,
        "health_after": health_after,
        "example_id": datum.example_id,
        "supervised_token_count": datum.supervised_token_count,
        "loss": result.loss,
        "gradient_norm": result.gradient_norm,
        "artifact": asdict(store.inspect(resource_id)),
        "responses": journal.records,
    }


def _run_load_phase(
    *,
    service_url: str,
    token: str,
    store: _S3ArtifactStore,
    sampling_resource_id: str,
) -> dict[str, object]:
    """Load the exported adapter into a fresh GPU process and render a WMO datum."""
    journal = _ResponseJournal()
    started = time.monotonic()
    sampling_url = store.download_url(sampling_resource_id).get_secret_value()
    with _client(service_url, token, journal) as client:
        health_before = client.get("/healthz").json()
        response = client.post(
            "/v1/sessions",
            json={
                "base_model": MODEL,
                "model_revision": MODEL_REVISION,
                "lora_rank": _spec().lora_rank,
                "seed": _spec().seed,
                "maximum_datum_tokens": _spec().maximum_datum_tokens,
                "sampling_download_url": sampling_url,
            },
        )
        response.raise_for_status()
        session_id = response.json()["session_id"]
        session = AnyCloudTrainerSession(
            client=client,
            artifact_store=store,
            session_id=session_id,
        )
        (datum,) = session.render_examples((_example(),))
        health_after = client.get("/healthz").json()
    return {
        "phase": "load",
        "started_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "service_url": service_url,
        "health_before": health_before,
        "health_after": health_after,
        "sampling_artifact": asdict(store.inspect(sampling_resource_id)),
        "loaded": True,
        "rendered_example_id": datum.example_id,
        "supervised_token_count": datum.supervised_token_count,
        "responses": journal.records,
    }


def _arguments() -> argparse.Namespace:
    """Parse one explicit validation phase and its caller-owned resources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("initial", "resume", "load"), required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--bucket-region", default="us-west-2")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--resume-resource-id")
    parser.add_argument("--sampling-resource-id")
    return parser.parse_args()


def main() -> None:
    """Run one restart-separated phase and print secret-free JSON evidence."""
    arguments = _arguments()
    token = os.environ.get("TRAINER_TOKEN")
    if not token:
        raise RuntimeError("TRAINER_TOKEN must be set")
    store = _S3ArtifactStore(
        bucket=arguments.bucket,
        prefix=arguments.prefix,
        region=arguments.bucket_region,
    )
    if arguments.phase == "initial":
        result = _run_training_phase(
            phase="initial",
            service_url=arguments.service_url,
            token=token,
            store=store,
            resume_resource_id=None,
        )
    elif arguments.phase == "resume":
        if not arguments.resume_resource_id:
            raise ValueError("--resume-resource-id is required for the resume phase")
        result = _run_training_phase(
            phase="resume",
            service_url=arguments.service_url,
            token=token,
            store=store,
            resume_resource_id=arguments.resume_resource_id,
        )
    else:
        if not arguments.sampling_resource_id:
            raise ValueError("--sampling-resource-id is required for the load phase")
        result = _run_load_phase(
            service_url=arguments.service_url,
            token=token,
            store=store,
            sampling_resource_id=arguments.sampling_resource_id,
        )
    result.update(
        {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "wmo_upstream_revision": WMO_UPSTREAM_REVISION,
            "wmo_adapter_revision": WMO_ADAPTER_REVISION,
        }
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
