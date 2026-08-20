"""Process entrypoint for the production GPU runtime."""

import os

import uvicorn

from anycloud_trainer.app import create_app
from anycloud_trainer.runtime import HuggingFaceTrainerRuntime


def main() -> None:
    """Start the authenticated trainer on AnyCloud's assigned service port."""
    token = os.environ.get("TRAINER_TOKEN")
    if not token:
        raise RuntimeError("TRAINER_TOKEN must be set")
    port = int(os.environ.get("PORT", "8088"))
    uvicorn.run(
        create_app(HuggingFaceTrainerRuntime(), token=token),
        host="0.0.0.0",
        port=port,
        access_log=True,
    )


if __name__ == "__main__":
    main()
