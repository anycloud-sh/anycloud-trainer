"""HTTP contract tests over an injected deterministic trainer runtime."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anycloud_trainer.app import create_app
from anycloud_trainer.contracts import (
    HealthResponse,
    OpenSessionRequest,
    OpenSessionResponse,
    RenderedDatum,
    RenderExamplesRequest,
    RenderExamplesResponse,
    SaveArtifactRequest,
    SaveArtifactResponse,
    TrainBatchRequest,
    TrainBatchResponse,
)
from anycloud_trainer.errors import TrainerServiceError


class _FakeRuntime:
    """Record validated calls without importing a GPU framework."""

    def __init__(self) -> None:
        """Initialize empty request journals."""
        self.opens: list[OpenSessionRequest] = []
        self.renders: list[tuple[str, RenderExamplesRequest]] = []
        self.batches: list[tuple[str, TrainBatchRequest]] = []
        self.saves: list[tuple[str, SaveArtifactRequest]] = []

    def health(self) -> HealthResponse:
        """Return deterministic healthy CUDA facts."""
        return HealthResponse(
            cuda_available=True,
            cuda_verified=True,
            active_session=bool(self.opens),
        )

    def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
        """Record settings and return one fixed session identity."""
        self.opens.append(request)
        return OpenSessionResponse(session_id="session-1")

    def render_examples(
        self, session_id: str, request: RenderExamplesRequest
    ) -> RenderExamplesResponse:
        """Record examples and return one datum per example."""
        self.renders.append((session_id, request))
        return RenderExamplesResponse(
            datums=tuple(
                RenderedDatum(
                    datum_id=f"datum-{index}",
                    example_id=example.example_id,
                    supervised_token_count=7,
                )
                for index, example in enumerate(request.examples)
            )
        )

    def train_batch(self, session_id: str, request: TrainBatchRequest) -> TrainBatchResponse:
        """Record one update and return fixed finite metrics."""
        self.batches.append((session_id, request))
        return TrainBatchResponse(
            loss=0.5,
            gradient_norm=1.25,
            input_token_count=12,
            supervised_token_count=7,
        )

    def save_artifact(self, session_id: str, request: SaveArtifactRequest) -> SaveArtifactResponse:
        """Record one upload target and return a fixed digest."""
        self.saves.append((session_id, request))
        return SaveArtifactResponse(sha256="a" * 64, size_bytes=123)


def _client(runtime: _FakeRuntime | None = None) -> tuple[TestClient, _FakeRuntime]:
    """Build one authenticated test client and its runtime journal."""
    selected = runtime or _FakeRuntime()
    return TestClient(create_app(selected, token="test-token")), selected


def _headers() -> dict[str, str]:
    """Return the valid test authorization header."""
    return {"Authorization": "Bearer test-token"}


def test_health_is_public_but_training_requires_authorization() -> None:
    """Expose liveness without exposing any mutation endpoint."""
    client, _runtime = _client()

    assert client.get("/healthz").json() == {
        "status": "ok",
        "cuda_available": True,
        "cuda_verified": True,
        "active_session": False,
    }
    response = client.post(
        "/v1/sessions",
        json={"base_model": "Qwen/Qwen3.5-4B", "lora_rank": 8, "seed": 0},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid trainer authorization"}


def test_complete_session_contract_preserves_order_and_upload_target() -> None:
    """Drive open, render, train, and save through their public wire shapes."""
    client, runtime = _client()
    opened = client.post(
        "/v1/sessions",
        headers=_headers(),
        json={
            "base_model": "Qwen/Qwen3.5-4B",
            "model_revision": "b" * 40,
            "lora_rank": 8,
            "seed": 7,
            "maximum_datum_tokens": 128,
        },
    )
    assert opened.status_code == 200
    assert opened.json() == {"session_id": "session-1"}

    rendered = client.post(
        "/v1/sessions/session-1/datums:render",
        headers=_headers(),
        json={
            "examples": [
                {
                    "example_id": "example-1",
                    "messages": [
                        {"role": "system", "content": "Task:\nAnswer briefly."},
                        {"role": "assistant", "content": "Done."},
                    ],
                }
            ]
        },
    )
    assert rendered.status_code == 200
    assert rendered.json() == {
        "datums": [
            {
                "datum_id": "datum-0",
                "example_id": "example-1",
                "supervised_token_count": 7,
            }
        ]
    }

    trained = client.post(
        "/v1/sessions/session-1/batches:train",
        headers=_headers(),
        json={"datum_ids": ["datum-0"], "learning_rate": 0.0001},
    )
    assert trained.status_code == 200
    assert trained.json()["loss"] == 0.5

    saved = client.post(
        "/v1/sessions/session-1/artifacts:save",
        headers=_headers(),
        json={
            "kind": "state",
            "name": "checkpoint-1",
            "upload_url": "https://objects.example/upload?signature=secret",
        },
    )
    assert saved.status_code == 200
    assert saved.json() == {"sha256": "a" * 64, "size_bytes": 123}
    assert runtime.opens[0].base_model == "Qwen/Qwen3.5-4B"
    assert runtime.opens[0].model_revision == "b" * 40
    assert runtime.renders[0][1].examples[0].example_id == "example-1"
    assert runtime.batches[0][1].datum_ids == ("datum-0",)
    assert runtime.saves[0][1].kind == "state"


def test_runtime_errors_are_safe_http_details() -> None:
    """Return only the runtime's public failure message."""

    class _FailingRuntime(_FakeRuntime):
        """Reject session creation with one declared conflict."""

        def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
            """Raise the public conflict without retaining the request."""
            raise TrainerServiceError("a training session is already active", status_code=409)

    client, _runtime = _client(_FailingRuntime())
    response = client.post(
        "/v1/sessions",
        headers=_headers(),
        json={"base_model": "Qwen/Qwen3.5-4B", "lora_rank": 8, "seed": 0},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "a training session is already active"}


def test_wire_rejects_non_assistant_target_and_duplicate_examples() -> None:
    """Stop malformed supervision before the runtime sees it."""
    client, runtime = _client()
    response = client.post(
        "/v1/sessions/session-1/datums:render",
        headers=_headers(),
        json={
            "examples": [
                {
                    "example_id": "duplicate",
                    "messages": [
                        {"role": "system", "content": "Task"},
                        {"role": "user", "content": "Not a target"},
                    ],
                },
                {
                    "example_id": "duplicate",
                    "messages": [
                        {"role": "system", "content": "Task"},
                        {"role": "assistant", "content": "Target"},
                    ],
                },
            ]
        },
    )

    assert response.status_code == 422
    assert runtime.renders == []


def test_wire_rejects_nonfinite_learning_rate_and_unencrypted_artifact_urls() -> None:
    """Stop unsafe numeric and transfer settings before the runtime sees them."""
    client, runtime = _client()
    trained = client.post(
        "/v1/sessions/session-1/batches:train",
        headers=_headers(),
        json={"datum_ids": ["datum-0"], "learning_rate": "Infinity"},
    )
    saved = client.post(
        "/v1/sessions/session-1/artifacts:save",
        headers=_headers(),
        json={
            "kind": "state",
            "name": "checkpoint-1",
            "upload_url": "http://objects.example/upload",
        },
    )

    assert trained.status_code == 422
    assert saved.status_code == 422
    assert runtime.batches == []
    assert runtime.saves == []
