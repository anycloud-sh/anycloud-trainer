"""Single-session Hugging Face LoRA runtime for an AnyCloud GPU Service."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

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
from anycloud_trainer.rendering import render_final_assistant

_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class _Datum:
    """Tokenized input and target mask retained outside GPU memory."""

    datum_id: str
    example_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_token_count: int


@dataclass
class _Session:
    """One exclusive model, optimizer, tokenizer, and rendered-datum collection."""

    session_id: str
    request: OpenSessionRequest
    tokenizer: Any
    model: Any
    optimizer: Any
    datums: dict[str, _Datum] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class HuggingFaceTrainerRuntime:
    """Execute one stateful LoRA session on the first visible CUDA device."""

    def __init__(self) -> None:
        """Initialize an empty exclusive-session slot."""
        self._session: _Session | None = None
        self._lifecycle_lock = threading.Lock()
        self._cuda_verified = _verify_cuda_execution()

    def health(self) -> HealthResponse:
        """Return process liveness and CUDA availability."""
        return HealthResponse(
            cuda_available=torch.cuda.is_available(),
            cuda_verified=self._cuda_verified,
            active_session=self._session is not None,
        )

    def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
        """Load one base model, attach LoRA weights, and optionally restore state."""
        if not torch.cuda.is_available():
            raise TrainerServiceError("CUDA is unavailable", status_code=503)
        with self._lifecycle_lock:
            if self._session is not None:
                raise TrainerServiceError("a training session is already active", status_code=409)
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            tokenizer = AutoTokenizer.from_pretrained(
                request.base_model,
                revision=request.model_revision,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            configuration = AutoConfig.from_pretrained(
                request.base_model,
                revision=request.model_revision,
            )
            model_factory = (
                AutoModelForImageTextToText
                if hasattr(configuration, "vision_config")
                else AutoModelForCausalLM
            )
            model = model_factory.from_pretrained(
                request.base_model,
                revision=request.model_revision,
                config=configuration,
                dtype=torch.bfloat16,
                device_map={"": 0},
                attn_implementation="sdpa",
            )
            text_configuration = getattr(model.config, "text_config", model.config)
            text_configuration.use_cache = False
            model = get_peft_model(
                model,
                LoraConfig(
                    r=request.lora_rank,
                    lora_alpha=request.lora_rank * 2,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                    exclude_modules=r".*visual.*",
                    revision=request.model_revision,
                ),
            )
            model.train()
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-4,
            )
            if request.resume_download_url is not None:
                self._restore_state(
                    str(request.resume_download_url),
                    request=request,
                    model=model,
                    optimizer=optimizer,
                )
            elif request.sampling_download_url is not None:
                self._restore_sampling(
                    str(request.sampling_download_url),
                    request=request,
                    model=model,
                )
            session_id = uuid.uuid4().hex
            self._session = _Session(
                session_id=session_id,
                request=request,
                tokenizer=tokenizer,
                model=model,
                optimizer=optimizer,
            )
            return OpenSessionResponse(session_id=session_id)

    def render_examples(
        self, session_id: str, request: RenderExamplesRequest
    ) -> RenderExamplesResponse:
        """Render final-assistant supervision while trimming context only."""
        session = self._require_session(session_id)
        rendered: list[RenderedDatum] = []
        with session.lock:
            for example in request.examples:
                rendered_tokens = render_final_assistant(
                    session.tokenizer,
                    example.messages,
                    maximum_tokens=session.request.maximum_datum_tokens,
                    example_id=example.example_id,
                )
                datum_id = hashlib.sha256(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "example_id": example.example_id,
                            "tokens": rendered_tokens.input_ids,
                            "labels": rendered_tokens.labels,
                        },
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                datum = _Datum(
                    datum_id=datum_id,
                    example_id=example.example_id,
                    input_ids=rendered_tokens.input_ids,
                    labels=rendered_tokens.labels,
                    supervised_token_count=rendered_tokens.supervised_token_count,
                )
                session.datums[datum_id] = datum
                rendered.append(
                    RenderedDatum(
                        datum_id=datum_id,
                        example_id=example.example_id,
                        supervised_token_count=rendered_tokens.supervised_token_count,
                    )
                )
        return RenderExamplesResponse(datums=tuple(rendered))

    def train_batch(self, session_id: str, request: TrainBatchRequest) -> TrainBatchResponse:
        """Perform one forward, backward, and AdamW update over retained datums."""
        session = self._require_session(session_id)
        with session.lock:
            try:
                datums = [session.datums[datum_id] for datum_id in request.datum_ids]
            except KeyError as exc:
                raise TrainerServiceError("the batch names an unknown rendered datum") from exc
            maximum_length = max(len(datum.input_ids) for datum in datums)
            pad_token_id = int(session.tokenizer.pad_token_id)
            input_rows: list[list[int]] = []
            label_rows: list[list[int]] = []
            attention_rows: list[list[int]] = []
            for datum in datums:
                padding = maximum_length - len(datum.input_ids)
                input_rows.append([*datum.input_ids, *([pad_token_id] * padding)])
                label_rows.append([*datum.labels, *([-100] * padding)])
                attention_rows.append([*([1] * len(datum.input_ids)), *([0] * padding)])
            input_ids = torch.tensor(input_rows, dtype=torch.long, device="cuda")
            labels = torch.tensor(label_rows, dtype=torch.long, device="cuda")
            attention_mask = torch.tensor(attention_rows, dtype=torch.long, device="cuda")
            for group in session.optimizer.param_groups:
                group["lr"] = request.learning_rate
            session.optimizer.zero_grad(set_to_none=True)
            output = session.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise TrainerServiceError("the training step produced a non-finite loss")
            loss.backward()
            gradient_norm = _gradient_norm(session.model)
            if not math.isfinite(gradient_norm):
                raise TrainerServiceError("the training step produced a non-finite gradient norm")
            session.optimizer.step()
            torch.cuda.synchronize()
            return TrainBatchResponse(
                loss=float(loss.detach().cpu()),
                gradient_norm=gradient_norm,
                input_token_count=sum(len(datum.input_ids) for datum in datums),
                supervised_token_count=sum(datum.supervised_token_count for datum in datums),
            )

    def save_artifact(self, session_id: str, request: SaveArtifactRequest) -> SaveArtifactResponse:
        """Serialize and upload resumable state or final PEFT sampling weights."""
        session = self._require_session(session_id)
        with session.lock, tempfile.TemporaryDirectory(prefix="anycloud-trainer-") as directory:
            root = Path(directory)
            if request.kind == "state":
                artifact_path = root / "trainer-state.pt"
                torch.save(
                    {
                        "base_model": session.request.base_model,
                        "model_revision": session.request.model_revision,
                        "lora_rank": session.request.lora_rank,
                        "seed": session.request.seed,
                        "adapter_state": _cpu_tree(get_peft_model_state_dict(session.model)),
                        "optimizer_state": _cpu_tree(session.optimizer.state_dict()),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                    artifact_path,
                )
            else:
                adapter_directory = root / "adapter"
                session.model.save_pretrained(adapter_directory, safe_serialization=True)
                session.tokenizer.save_pretrained(adapter_directory)
                artifact_path = root / "sampling-adapter.tar.gz"
                with tarfile.open(artifact_path, "w:gz") as archive:
                    archive.add(adapter_directory, arcname="adapter")
            digest = _sha256_file(artifact_path)
            size_bytes = artifact_path.stat().st_size
            try:
                with artifact_path.open("rb") as payload:
                    response = httpx.put(
                        str(request.upload_url),
                        content=iter(lambda: payload.read(1024 * 1024), b""),
                        headers={
                            "Content-Length": str(size_bytes),
                            "Content-Type": "application/octet-stream",
                        },
                        timeout=900,
                    )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise TrainerServiceError(
                    "the trainer artifact upload failed", status_code=502
                ) from exc
            return SaveArtifactResponse(sha256=digest, size_bytes=size_bytes)

    def _require_session(self, session_id: str) -> _Session:
        """Return the live session only when the exact opaque identity matches."""
        session = self._session
        if session is None or session.session_id != session_id:
            raise TrainerServiceError("training session not found", status_code=404)
        return session

    def _restore_state(
        self,
        download_url: str,
        *,
        request: OpenSessionRequest,
        model: Any,
        optimizer: Any,
    ) -> None:
        """Download and verify one state artifact before applying any weight change."""
        with tempfile.NamedTemporaryFile(prefix="anycloud-state-", suffix=".pt") as state_file:
            _download_artifact(download_url, Path(state_file.name), kind="state")
            try:
                state = torch.load(state_file.name, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise TrainerServiceError("the trainer state is invalid") from exc
        if not isinstance(state, dict):
            raise TrainerServiceError("the trainer state has an invalid root value")
        expected = (
            request.base_model,
            request.model_revision,
            request.lora_rank,
            request.seed,
        )
        observed = (
            state.get("base_model"),
            state.get("model_revision"),
            state.get("lora_rank"),
            state.get("seed"),
        )
        if observed != expected:
            raise TrainerServiceError(
                "the trainer state does not match the requested model settings"
            )
        required = (
            "adapter_state",
            "optimizer_state",
            "torch_rng_state",
            "cuda_rng_state_all",
        )
        if any(key not in state for key in required):
            raise TrainerServiceError("the trainer state is missing required values")
        try:
            set_peft_model_state_dict(model, state["adapter_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainerServiceError("the trainer state could not be restored") from exc
        _move_optimizer_state(optimizer, device=torch.device("cuda"))

    def _restore_sampling(
        self,
        download_url: str,
        *,
        request: OpenSessionRequest,
        model: Any,
    ) -> None:
        """Load one exported standard PEFT adapter into a fresh base model."""
        with tempfile.TemporaryDirectory(prefix="anycloud-sampling-") as directory:
            root = Path(directory)
            archive_path = root / "sampling-adapter.tar.gz"
            _download_artifact(download_url, archive_path, kind="sampling adapter")
            extraction_root = root / "extracted"
            extraction_root.mkdir()
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    _extract_regular_files(archive, extraction_root)
            except (OSError, tarfile.TarError) as exc:
                raise TrainerServiceError("the sampling adapter archive is invalid") from exc
            adapter_root = extraction_root / "adapter"
            config_path = adapter_root / "adapter_config.json"
            weights_path = adapter_root / "adapter_model.safetensors"
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TrainerServiceError("the sampling adapter config is invalid") from exc
            if not isinstance(config, dict):
                raise TrainerServiceError("the sampling adapter config is invalid")
            expected = (request.base_model, request.model_revision, request.lora_rank)
            observed = (
                config.get("base_model_name_or_path"),
                config.get("revision"),
                config.get("r"),
            )
            if observed != expected:
                raise TrainerServiceError(
                    "the sampling adapter does not match the requested model settings"
                )
            try:
                adapter_state = load_safetensors(weights_path, device="cpu")
                set_peft_model_state_dict(model, adapter_state)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise TrainerServiceError("the sampling adapter weights are invalid") from exc


def _verify_cuda_execution() -> bool:
    """Prove that the visible accelerator can allocate, launch work, and synchronize."""
    if not torch.cuda.is_available():
        return False
    try:
        left = torch.tensor([2.0], device="cuda")
        result = left * 3.0
        torch.cuda.synchronize()
        return float(result.cpu()[0]) == 6.0
    except RuntimeError:
        return False


def _download_artifact(url: str, path: Path, *, kind: str) -> None:
    """Stream one bounded caller-authorized artifact without retaining its URL."""
    try:
        with httpx.stream("GET", url, timeout=900) as response:
            response.raise_for_status()
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) > _MAX_ARTIFACT_BYTES:
                raise TrainerServiceError(f"the trainer {kind} exceeds the size limit")
            size_bytes = 0
            with path.open("wb") as destination:
                for chunk in response.iter_bytes(1024 * 1024):
                    size_bytes += len(chunk)
                    if size_bytes > _MAX_ARTIFACT_BYTES:
                        raise TrainerServiceError(f"the trainer {kind} exceeds the size limit")
                    destination.write(chunk)
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise TrainerServiceError(f"the trainer {kind} download failed", status_code=502) from exc


def _extract_regular_files(archive: tarfile.TarFile, root: Path) -> None:
    """Extract only bounded regular files beneath the dedicated temporary root."""
    resolved_root = root.resolve()
    members = archive.getmembers()
    if len(members) > 10_000 or sum(member.size for member in members) > _MAX_ARTIFACT_BYTES:
        raise TrainerServiceError("the sampling adapter archive exceeds the extraction limit")
    for member in members:
        destination = (root / member.name).resolve()
        if resolved_root != destination and resolved_root not in destination.parents:
            raise TrainerServiceError("the sampling adapter archive contains an unsafe path")
        if not member.isfile() and not member.isdir():
            raise TrainerServiceError("the sampling adapter archive contains an unsafe member")
        archive.extract(member, path=root)


def _gradient_norm(model: Any) -> float:
    """Compute the finite global L2 norm without clipping gradients."""
    squared_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm(2).cpu())
        squared_norm += norm * norm
    return math.sqrt(squared_norm)


def _cpu_tree(value: Any) -> Any:
    """Move every tensor in a nested optimizer or adapter state onto CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _move_optimizer_state(optimizer: Any, *, device: torch.device) -> None:
    """Move restored optimizer tensors to the active model device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
