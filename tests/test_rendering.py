"""Final-assistant rendering tests independent of GPU libraries and model downloads."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from anycloud_trainer.contracts import ConversationMessage
from anycloud_trainer.errors import TrainerServiceError
from anycloud_trainer.rendering import render_final_assistant


class _BoundaryMergingTokenizer:
    """Emulate a tokenizer that merges one context and one target newline."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Return a prefix or full text whose boundary crosses one token."""
        del tokenize
        if add_generation_prompt:
            return "context\n"
        assert len(conversation) == 2
        return "context\n\ntarget"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, object]:
        """Return four deterministic IDs and offsets over the full rendered text."""
        assert text == "context\n\ntarget"
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [10, 11, 12, 13],
            "offset_mapping": [(0, 7), (7, 9), (9, 12), (12, 15)],
        }


def _messages() -> tuple[ConversationMessage, ...]:
    """Return one minimal context and final target pair."""
    return (
        ConversationMessage(role="user", content="context"),
        ConversationMessage(role="assistant", content="target"),
    )


def test_boundary_crossing_token_is_masked_instead_of_training_on_context() -> None:
    """Only tokens starting wholly after the character boundary receive labels."""
    result = render_final_assistant(
        _BoundaryMergingTokenizer(),
        _messages(),
        maximum_tokens=None,
        example_id="example-1",
    )

    assert result.input_ids == (10, 11, 12, 13)
    assert result.labels == (-100, -100, 12, 13)
    assert result.supervised_token_count == 2


def test_context_only_truncation_retains_the_complete_supervised_target() -> None:
    """The token ceiling removes context but never a final-assistant target token."""
    result = render_final_assistant(
        _BoundaryMergingTokenizer(),
        _messages(),
        maximum_tokens=2,
        example_id="example-1",
    )

    assert result.input_ids == (12, 13)
    assert result.labels == (12, 13)
    assert result.supervised_token_count == 2

    with pytest.raises(TrainerServiceError, match="cannot retain the complete supervised target"):
        render_final_assistant(
            _BoundaryMergingTokenizer(),
            _messages(),
            maximum_tokens=1,
            example_id="example-1",
        )
