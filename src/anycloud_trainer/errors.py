"""Safe service exceptions mapped to bounded HTTP responses."""


class TrainerServiceError(RuntimeError):
    """A request cannot complete without exposing internal or credential detail."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        """Store one public message and its intended HTTP status."""
        super().__init__(message)
        self.status_code = status_code
