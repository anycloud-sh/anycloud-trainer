"""Tokenizer-only final-assistant supervision with context-safe truncation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from anycloud_trainer.contracts import ConversationMessage
from anycloud_trainer.errors import TrainerServiceError


class ChatTemplateTokenizer(Protocol):
    """Small tokenizer surface needed to identify the final assistant boundary."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: Literal[False],
        add_generation_prompt: bool,
    ) -> str:
        """Render a conversation to its exact model-facing text."""
        ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, object]:
        """Tokenize text while returning character offsets for each token."""
        ...


@dataclass(frozen=True)
class RenderedTrainingTokens:
    """One tokenized datum with only final-assistant tokens supervised."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_token_count: int


def render_final_assistant(
    tokenizer: ChatTemplateTokenizer,
    messages: Sequence[ConversationMessage],
    *,
    maximum_tokens: int | None,
    example_id: str,
) -> RenderedTrainingTokens:
    """Render one conversation and mask every token touching caller context.

    A tokenizer can merge characters across the prompt/target boundary. Character offsets let us
    mask that crossing token rather than accidentally training on caller context. The next token
    begins final-assistant supervision.
    """
    payload = [_message_payload(message) for message in messages]
    full_text = tokenizer.apply_chat_template(
        payload,
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        payload[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(full_text, str) or not isinstance(prefix_text, str):
        raise TrainerServiceError("the model chat template returned unsupported text")
    if not full_text.startswith(prefix_text):
        raise TrainerServiceError(
            f"the model chat template did not preserve the target boundary for {example_id}"
        )
    tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _integer_sequence(tokenized.get("input_ids"), label="input_ids")
    offsets = _offset_sequence(tokenized.get("offset_mapping"))
    if len(input_ids) != len(offsets):
        raise TrainerServiceError("the model tokenizer returned mismatched token offsets")
    boundary = len(prefix_text)
    prefix_length = next(
        (index for index, (start, end) in enumerate(offsets) if start >= boundary and end > start),
        len(input_ids),
    )
    if maximum_tokens is not None and len(input_ids) > maximum_tokens:
        trim_count = len(input_ids) - maximum_tokens
        if trim_count > prefix_length:
            raise TrainerServiceError(
                "maximum_datum_tokens cannot retain the complete supervised target "
                f"for {example_id}"
            )
        input_ids = input_ids[trim_count:]
        prefix_length -= trim_count
    supervised_token_count = len(input_ids) - prefix_length
    if supervised_token_count <= 0:
        raise TrainerServiceError(
            f"the model chat template produced no supervised tokens for {example_id}"
        )
    labels = (-100,) * prefix_length + tuple(input_ids[prefix_length:])
    return RenderedTrainingTokens(
        input_ids=tuple(input_ids),
        labels=labels,
        supervised_token_count=supervised_token_count,
    )


def _message_payload(message: ConversationMessage) -> dict[str, object]:
    """Convert one validated message into the sparse mapping expected by chat templates."""
    payload: dict[str, object] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [call.model_dump(mode="json") for call in message.tool_calls]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _integer_sequence(value: object, *, label: str) -> list[int]:
    """Validate one tokenizer sequence without coercing booleans or numeric strings."""
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TrainerServiceError(f"the model tokenizer returned unsupported {label}")
    return value


def _offset_sequence(value: object) -> list[tuple[int, int]]:
    """Validate one ordered tokenizer character-offset sequence."""
    if not isinstance(value, list):
        raise TrainerServiceError("the model tokenizer returned unsupported offsets")
    offsets: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or type(item[0]) is not int
            or type(item[1]) is not int
            or item[0] < 0
            or item[1] < item[0]
        ):
            raise TrainerServiceError("the model tokenizer returned unsupported offsets")
        offsets.append((item[0], item[1]))
    return offsets
