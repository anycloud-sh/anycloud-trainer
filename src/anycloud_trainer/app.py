"""Authenticated FastAPI surface for one stateful GPU trainer."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from anycloud_trainer.contracts import (
    HealthResponse,
    OpenSessionRequest,
    OpenSessionResponse,
    RenderExamplesRequest,
    RenderExamplesResponse,
    SaveArtifactRequest,
    SaveArtifactResponse,
    TrainBatchRequest,
    TrainBatchResponse,
    TrainerRuntime,
)
from anycloud_trainer.errors import TrainerServiceError


def _authorization_dependency(token: str) -> Callable[[str | None], None]:
    """Build a constant-time bearer-token check without retaining request values."""
    expected = f"Bearer {token}"

    def require_authorization(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        """Reject a missing or incorrect service credential."""
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid trainer authorization")

    return require_authorization


def create_app(runtime: TrainerRuntime, *, token: str) -> FastAPI:
    """Create the trainer application around an injected runtime.

    Args:
        runtime: Stateful GPU runtime or deterministic test implementation.
        token: Non-empty bearer credential injected into the service process.

    Returns:
        A configured FastAPI application.

    Raises:
        ValueError: The service token is empty.
    """
    if not token:
        raise ValueError("trainer token must be non-empty")
    app = FastAPI(title="AnyCloud Trainer", version="0.1.0")
    authorize = _authorization_dependency(token)

    @app.exception_handler(TrainerServiceError)
    async def trainer_error_handler(_request: Request, exc: TrainerServiceError) -> JSONResponse:
        """Map safe runtime failures to their declared status codes."""
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return liveness and CUDA availability without authentication."""
        return runtime.health()

    @app.post(
        "/v1/sessions",
        response_model=OpenSessionResponse,
        dependencies=[Depends(authorize)],
    )
    def open_session(request: OpenSessionRequest) -> OpenSessionResponse:
        """Create or restore one exclusive LoRA session."""
        return runtime.open_session(request)

    @app.post(
        "/v1/sessions/{session_id}/datums:render",
        response_model=RenderExamplesResponse,
        dependencies=[Depends(authorize)],
    )
    def render_examples(session_id: str, request: RenderExamplesRequest) -> RenderExamplesResponse:
        """Render and retain an ordered set of supervised datums."""
        return runtime.render_examples(session_id, request)

    @app.post(
        "/v1/sessions/{session_id}/batches:train",
        response_model=TrainBatchResponse,
        dependencies=[Depends(authorize)],
    )
    def train_batch(session_id: str, request: TrainBatchRequest) -> TrainBatchResponse:
        """Complete exactly one optimizer update."""
        return runtime.train_batch(session_id, request)

    @app.post(
        "/v1/sessions/{session_id}/artifacts:save",
        response_model=SaveArtifactResponse,
        dependencies=[Depends(authorize)],
    )
    def save_artifact(session_id: str, request: SaveArtifactRequest) -> SaveArtifactResponse:
        """Upload one immutable state or sampling artifact."""
        return runtime.save_artifact(session_id, request)

    return app
