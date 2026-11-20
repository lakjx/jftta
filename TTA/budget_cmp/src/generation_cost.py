from __future__ import annotations
import inspect
import json
import os
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator
import torch

_ACTIVE_ATTRIBUTE = "_tta_generation_cost_meter"
_MISSING = object()


def recording_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("runtime", {}).get("record_generation_cost", False))


def generated_token_counts(
    sequences: torch.Tensor,
    input_width: int,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    pad_token_id: int | None,
) -> list[int]:
    if sequences.ndim != 2 or input_width < 0 or input_width > sequences.shape[1]:
        raise ValueError(
            "Expected decoder-only generated sequences with their padded input width."
        )
    if isinstance(eos_token_id, torch.Tensor):
        eos_token_id = eos_token_id.detach().cpu().tolist()
    if isinstance(eos_token_id, int):
        stops = {eos_token_id}
    else:
        stops = set(eos_token_id or [])
    counts: list[int] = []
    for continuation in sequences[:, input_width:].detach().cpu().tolist():
        stop = next(
            (index for index, token in enumerate(continuation) if token in stops), None
        )
        if stop is not None:
            counts.append(stop + 1)
            continue
        length = len(continuation)
        while (
            length
            and pad_token_id is not None
            and (continuation[length - 1] == pad_token_id)
        ):
            length -= 1
        counts.append(length)
    return counts


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GenerationCostRecorder:
    def __init__(self, model: Any, device: torch.device, method: str):
        self.model = model
        self.device = device
        self.method = method
        self.stage = f"{method}.generation"
        self.stage_metadata: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.elapsed_seconds = 0.0
        self._started = 0.0
        self._original_generate: Any = None
        self._own_generate: Any = _MISSING

    def __enter__(self) -> GenerationCostRecorder:
        if getattr(self.model, _ACTIVE_ATTRIBUTE, None) is not None:
            raise RuntimeError(
                "A generation cost recorder is already active for this model."
            )
        self._original_generate = self.model.generate
        self._own_generate = vars(self.model).get("generate", _MISSING)
        setattr(self.model, _ACTIVE_ATTRIBUTE, self)
        self.model.generate = self._generate
        try:
            _synchronize(self.device)
            self._started = time.monotonic()
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc is None:
                _synchronize(self.device)
            self.elapsed_seconds = time.monotonic() - self._started
        finally:
            self._restore()

    def _restore(self) -> None:
        if self._own_generate is _MISSING:
            delattr(self.model, "generate")
        else:
            self.model.generate = self._own_generate
        delattr(self.model, _ACTIVE_ATTRIBUTE)

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("input_ids", kwargs.get("inputs"))
        if inputs is None and args:
            inputs = args[0]
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError(
                "Generation cost recording requires decoder-only input_ids."
            )
        config = kwargs.get(
            "generation_config", getattr(self.model, "generation_config", None)
        )
        eos = kwargs.get("eos_token_id", getattr(config, "eos_token_id", None))
        pad = kwargs.get("pad_token_id", getattr(config, "pad_token_id", None))
        call: dict[str, Any] = {
            "call_index": len(self.calls),
            "stage": self.stage,
            "stage_metadata": dict(self.stage_metadata),
            "input_sequences": int(inputs.shape[0]),
            "input_width": int(inputs.shape[1]),
            "num_return_sequences": int(kwargs.get("num_return_sequences", 1)),
            "max_new_tokens": kwargs.get("max_new_tokens"),
        }
        _synchronize(self.device)
        started = time.monotonic()
        try:
            outputs = self._original_generate(*args, **kwargs)
        except BaseException as exc:
            call.update(
                status="failed",
                error_type=type(exc).__name__,
                generated_token_counts=None,
                total_generated_tokens=None,
                elapsed_seconds=time.monotonic() - started,
            )
            self.calls.append(call)
            raise
        _synchronize(self.device)
        elapsed = time.monotonic() - started
        sequences = outputs if isinstance(outputs, torch.Tensor) else outputs.sequences
        counts = generated_token_counts(sequences, int(inputs.shape[1]), eos, pad)
        call.update(
            status="completed",
            generated_token_counts=counts,
            total_generated_tokens=sum(counts),
            elapsed_seconds=elapsed,
        )
        self.calls.append(call)
        return outputs

    def summary(self) -> dict[str, Any]:
        completed = [call for call in self.calls if call["status"] == "completed"]
        failed = [call for call in self.calls if call["status"] == "failed"]
        return {
            "schema_version": 1,
            "token_convention": "includes_first_eos_excludes_padding",
            "total_generated_tokens": sum(
                (call["total_generated_tokens"] for call in completed)
            ),
            "completed_call_count": len(completed),
            "generated_sequence_count": sum(
                (len(call["generated_token_counts"]) for call in completed)
            ),
            "completed_elapsed_seconds": sum(
                (call["elapsed_seconds"] for call in completed)
            ),
            "failed_call_count": len(failed),
            "failed_elapsed_seconds": sum((call["elapsed_seconds"] for call in failed)),
            "failed_generated_tokens": None if failed else 0,
            "token_count_complete": not failed,
            "elapsed_seconds": self.elapsed_seconds,
            "calls": self.calls,
        }

    def attach(self, row: dict[str, Any]) -> None:
        summary = self.summary()
        if "tta_generated_tokens" in row:
            row["tta_generated_tokens_legacy"] = row["tta_generated_tokens"]
        if "elapsed_seconds" in row:
            row["elapsed_seconds_legacy"] = row["elapsed_seconds"]
        row["generation_cost"] = summary
        row["tta_generated_tokens"] = summary["total_generated_tokens"]
        row["elapsed_seconds"] = summary["elapsed_seconds"]


@contextmanager
def generation_stage(model: Any, stage: str, **metadata: Any) -> Iterator[None]:
    recorder = getattr(model, _ACTIVE_ATTRIBUTE, None)
    if recorder is None:
        yield
        return
    previous = (recorder.stage, recorder.stage_metadata)
    recorder.stage, recorder.stage_metadata = (stage, metadata)
    try:
        yield
    finally:
        recorder.stage, recorder.stage_metadata = previous


def record_failed_prompt(
    cfg: dict[str, Any],
    prompt: dict[str, Any],
    method: str,
    recorder: GenerationCostRecorder,
    error: BaseException,
) -> None:
    output_dir = cfg.get("paths", {}).get("output_dir")
    if not output_dir:
        return
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    path = Path(output_dir) / f"generation_cost_failures.rank{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_id": prompt["prompt_id"],
        "method": method,
        "error_type": type(error).__name__,
        "attempt_finished_unix": time.time(),
        "pid": os.getpid(),
        "generation_cost": recorder.summary(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def record_prompt_cost(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        bound = signature.bind(*args, **kwargs)
        cfg = bound.arguments["cfg"]
        if not recording_enabled(cfg):
            return function(*args, **kwargs)
        bound.apply_defaults()
        model = bound.arguments["model"]
        device = bound.arguments["device"]
        prompt = bound.arguments["prompt"]
        method = str(
            bound.arguments.get(
                "method", cfg.get("method", {}).get("name", function.__name__)
            )
        )
        recorder = GenerationCostRecorder(model, device, method)
        try:
            with recorder:
                row = function(*args, **kwargs)
        except BaseException as exc:
            record_failed_prompt(cfg, prompt, method, recorder, exc)
            raise
        recorder.attach(row)
        return row

    return wrapped
