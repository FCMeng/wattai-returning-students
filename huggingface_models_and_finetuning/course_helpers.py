"""Transparent helpers for the standalone Hugging Face fine-tuning course.

The notebooks show the implementation of each important helper before importing
it from here.  Importing this module never downloads a model or dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COURSE_ROOT = Path(__file__).resolve().parent
CLASSROOM_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CLASSROOM_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
BASE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
EXTENDED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXTENDED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN35_MODEL = "Qwen/Qwen3.5-9B"
QWEN35_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
QWEN3_OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN3_OMNI_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
QWEN3_OMNI_THINKING_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
QWEN3_OMNI_THINKING_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"


def find_course_root(start: str | Path | None = None) -> Path:
    """Find the course root from the notebook or course directory."""
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / "course_helpers.py").is_file() and (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the course folder. Start JupyterLab inside the downloaded course folder."
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records and report a useful line number on failure."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"Line {line_number} of {path} is not a JSON object")
        rows.append(value)
    return rows


def sha256_file(path: str | Path) -> str:
    """Return a reproducible SHA-256 digest without loading the whole file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    """Reject malformed role/content sequences before tokenization."""
    if len(messages) < 2:
        raise ValueError("A training conversation needs at least a prompt and an answer.")
    allowed = {"system", "user", "assistant"}
    for index, message in enumerate(messages):
        if message.get("role") not in allowed:
            raise ValueError(f"Message {index} has unsupported role {message.get('role')!r}.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Message {index} needs non-empty text content.")
    if messages[-1]["role"] != "assistant":
        raise ValueError("The final message must be the assistant answer used as the target.")


def audit_splits(train_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check tree isolation, IDs, roles, and exact duplicate conversations."""
    for row in [*train_rows, *eval_rows]:
        validate_messages(row["messages"])
        if not row.get("source_tree_id"):
            raise ValueError(f"Record {row.get('id', '<missing id>')} has no source_tree_id")

    train_trees = {str(row["source_tree_id"]) for row in train_rows}
    eval_trees = {str(row["source_tree_id"]) for row in eval_rows}
    overlap = sorted(train_trees & eval_trees)
    if overlap:
        raise AssertionError(f"Conversation-tree leakage detected: {overlap[:5]}")

    def signature(row: Mapping[str, Any]) -> str:
        canonical = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    train_signatures = [signature(row) for row in train_rows]
    eval_signatures = [signature(row) for row in eval_rows]
    return {
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "train_trees": len(train_trees),
        "evaluation_trees": len(eval_trees),
        "tree_overlap": len(overlap),
        "duplicate_train_rows": sum(count - 1 for count in Counter(train_signatures).values() if count > 1),
        "cross_split_exact_duplicates": len(set(train_signatures) & set(eval_signatures)),
    }


def split_prompt_completion(messages: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Convert a conversation into TRL's conversational prompt/completion format."""
    validate_messages(messages)
    return {
        "prompt": [dict(message) for message in messages[:-1]],
        "completion": [dict(messages[-1])],
    }


def render_messages(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render structured messages with the tokenizer version used by the model."""
    return tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def chat_token_ids(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Return a plain ID list across Transformers chat-template return types."""
    encoded = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Expected one conversation, but received a batch.")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def build_final_answer_labels(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> dict[str, list[int]]:
    """Create IDs whose loss is active only on the final assistant answer.

    The prefix is rendered with an assistant-generation marker.  The complete
    conversation is rendered without adding a second generation marker.  This
    is intentionally a final-answer objective; multi-assistant-turn objectives
    should use a training chat template that exposes assistant masks.
    """
    validate_messages(messages)
    prompt_messages = [dict(message) for message in messages[:-1]]
    full_messages = [dict(message) for message in messages]
    prompt_ids = chat_token_ids(
        prompt_messages, tokenizer, add_generation_prompt=True
    )
    input_ids = chat_token_ids(
        full_messages, tokenizer, add_generation_prompt=False
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The tokenizer's prompt rendering is not a prefix of the completed conversation. "
            "Inspect the chat template before building labels."
        )
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    attention_mask = [1] * len(input_ids)
    if max_length is not None and len(input_ids) > max_length:
        # Keep the answer end, but make the loss of prompt context visible to the caller.
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        attention_mask = attention_mask[-max_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("Truncation removed every supervised answer token.")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def active_label_fraction(labels: Sequence[int]) -> float:
    """Return the fraction of token positions that contribute to SFT loss."""
    return sum(label != -100 for label in labels) / max(1, len(labels))


def count_trainable_parameters(model: Any) -> dict[str, int | float]:
    """Count all and trainable parameters after PEFT injection."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": 100.0 * trainable / max(1, total),
    }


def fixed_generate(model: Any, tokenizer: Any, messages: Sequence[Mapping[str, str]], *, max_new_tokens: int = 80) -> str:
    """Generate deterministically so model variants can be compared fairly."""
    import torch

    model.eval()
    encoded = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuation = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def load_text_model(model_id: str, revision: str, *, dtype: Any | None = None) -> tuple[Any, Any]:
    """Load a pinned tokenizer and causal LM, choosing an available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    chosen_dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=chosen_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def directory_size(path: str | Path) -> int:
    """Return the total bytes in an adapter or merged-model directory."""
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def write_adapter_manifest(
    output_path: str | Path,
    *,
    base_model: str,
    base_revision: str,
    data_manifest_sha256: str,
    training_arguments: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write the minimum lineage record needed to interpret an adapter."""
    payload = {
        "base_model": base_model,
        "base_revision": base_revision,
        "data_manifest_sha256": data_manifest_sha256,
        "training_arguments": dict(training_arguments),
        "metrics": dict(metrics),
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def gibibytes(byte_count: float) -> float:
    """Convert bytes to GiB (2**30 bytes), keeping units explicit."""
    return byte_count / 2**30


def full_finetuning_memory_estimate(parameter_count: int) -> dict[str, float]:
    """Estimate persistent mixed-precision AdamW allocations, excluding activations."""
    allocations = {
        "BF16 weights": parameter_count * 2,
        "BF16 gradients": parameter_count * 2,
        "FP32 master weights": parameter_count * 4,
        "FP32 Adam moments": parameter_count * 8,
    }
    return {name: gibibytes(value) for name, value in allocations.items()}


def assert_same_shape(left: Any, right: Any) -> None:
    """Raise an explicit error when reload or merge changes logit dimensions."""
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError(f"Shape changed from {tuple(left.shape)} to {tuple(right.shape)}")
