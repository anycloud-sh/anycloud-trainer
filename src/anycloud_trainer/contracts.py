"""Validated HTTP and runtime contracts for the trainer service."""

from __future__ import annotations

import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class ContractModel(BaseModel):
    """Base model that rejects undeclared wire fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FunctionCall(ContractModel):
    """One canonical function call carried by an assistant message."""

    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=1_000_000)


class ToolCall(ContractModel):
    """One OpenAI-shaped tool call."""

    type: Literal["function"] = "function"
    id: str = Field(min_length=1, max_length=256)
    function: FunctionCall


class ConversationMessage(ContractModel):
    """One text, assistant-tool, or tool-result conversation message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=2_000_000)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_role_fields(self) -> ConversationMessage:
        """Keep tool-only fields attached to the roles that define them."""
        if self.tool_calls and self.role != "assistant":
            raise ValueError("tool_calls are only valid on assistant messages")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("tool_call_id is only valid on tool messages")
        if self.name is not None and self.role != "tool":
            raise ValueError("name is only valid on tool messages")
        return self


class TrainingExample(ContractModel):
    """One supervised example whose final assistant message is the target."""

    example_id: str = Field(min_length=1, max_length=256)
    messages: tuple[ConversationMessage, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def require_final_assistant(self) -> TrainingExample:
        """Require the complete target action to be the final message."""
        if self.messages[-1].role != "assistant":
            raise ValueError("the final training message must have role assistant")
        return self


class OpenSessionRequest(ContractModel):
    """Settings needed to create or restore one LoRA training session."""

    base_model: str = Field(min_length=1, max_length=512)
    model_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    lora_rank: int = Field(gt=0, le=4096)
    seed: int = Field(ge=0, le=2**32 - 1)
    maximum_datum_tokens: int | None = Field(default=None, gt=1)
    resume_download_url: HttpUrl | None = None

    @field_validator("resume_download_url")
    @classmethod
    def require_https_resume_url(cls, value: HttpUrl | None) -> HttpUrl | None:
        """Allow artifact restore only through an encrypted transfer URL."""
        if value is not None and value.scheme != "https":
            raise ValueError("resume_download_url must use HTTPS")
        return value


class OpenSessionResponse(ContractModel):
    """Opaque identity of one live training session."""

    session_id: str = Field(min_length=1, max_length=128)


class RenderExamplesRequest(ContractModel):
    """A bounded ordered group of examples to render on the trainer."""

    examples: tuple[TrainingExample, ...] = Field(min_length=1, max_length=10_000)

    @field_validator("examples")
    @classmethod
    def require_unique_examples(
        cls, examples: tuple[TrainingExample, ...]
    ) -> tuple[TrainingExample, ...]:
        """Reject duplicate example identities before any tokenizer work."""
        identities = [example.example_id for example in examples]
        if len(identities) != len(set(identities)):
            raise ValueError("example_id values must be unique")
        return examples


class RenderedDatum(ContractModel):
    """Opaque rendered datum metadata retained by a live session."""

    datum_id: str = Field(min_length=1, max_length=128)
    example_id: str = Field(min_length=1, max_length=256)
    supervised_token_count: int = Field(gt=0)


class RenderExamplesResponse(ContractModel):
    """Rendered datum metadata in exact request order."""

    datums: tuple[RenderedDatum, ...] = Field(min_length=1)


class TrainBatchRequest(ContractModel):
    """One ordered optimizer update over previously rendered datums."""

    datum_ids: tuple[str, ...] = Field(min_length=1)
    learning_rate: float = Field(gt=0)

    @field_validator("learning_rate")
    @classmethod
    def require_finite_learning_rate(cls, value: float) -> float:
        """Reject nonfinite optimizer settings before GPU mutation."""
        if not math.isfinite(value):
            raise ValueError("learning_rate must be finite")
        return value


class TrainBatchResponse(ContractModel):
    """Finite observations from one completed optimizer update."""

    loss: float
    gradient_norm: float
    input_token_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)

    @field_validator("loss", "gradient_norm")
    @classmethod
    def require_finite_metric(cls, value: float) -> float:
        """Keep nonfinite accelerator results off the public wire."""
        if not math.isfinite(value):
            raise ValueError("training metrics must be finite")
        return value


class SaveArtifactRequest(ContractModel):
    """A short-lived upload target for one immutable trainer artifact."""

    kind: Literal["state", "sampling"]
    name: str = Field(min_length=1, max_length=512)
    upload_url: HttpUrl

    @field_validator("upload_url")
    @classmethod
    def require_https_upload_url(cls, value: HttpUrl) -> HttpUrl:
        """Allow artifact upload only through an encrypted transfer URL."""
        if value.scheme != "https":
            raise ValueError("upload_url must use HTTPS")
        return value


class SaveArtifactResponse(ContractModel):
    """Digest and size of bytes successfully uploaded by the trainer."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)


class HealthResponse(ContractModel):
    """Public liveness and accelerator facts."""

    status: Literal["ok"] = "ok"
    cuda_available: bool
    cuda_verified: bool
    active_session: bool


class TrainerRuntime(Protocol):
    """Stateful GPU behavior owned by the HTTP application."""

    def health(self) -> HealthResponse:
        """Return current process and accelerator liveness."""
        ...

    def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
        """Create or restore one exclusive training session."""
        ...

    def render_examples(
        self, session_id: str, request: RenderExamplesRequest
    ) -> RenderExamplesResponse:
        """Render and retain supervised examples for one session."""
        ...

    def train_batch(self, session_id: str, request: TrainBatchRequest) -> TrainBatchResponse:
        """Perform exactly one optimizer update."""
        ...

    def save_artifact(self, session_id: str, request: SaveArtifactRequest) -> SaveArtifactResponse:
        """Upload immutable optimizer state or final sampling weights."""
        ...
