"""Shared helpers for the live RCD LLM agents and video-analysis course.

The module avoids network access at import time.  Every actual request requires
an active RCD allocation, an API key, and either the Clemson network or CUVPN.
The notebooks introduce each important helper in-place before importing the
reusable implementation from this module.
"""

from __future__ import annotations

import ast
import base64
import getpass
import json
import mimetypes
import operator
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


RCD_LLM_BASE_URL = "https://llm.rcd.clemson.edu/v1"
RCD_OPENAI_BASE_URL = "https://llm.rcd.clemson.edu/openai/v1"
COMPLETION_MODEL = "gemma-4-31b-non-it"
CHAT_MODEL = "qwen3-30b-a3b-instruct-fp8"
THINKING_MODEL = "qwen3.5-9b"
TOOL_MODEL = CHAT_MODEL
VIDEO_MODEL = "qwen3-omni-30b-a3b"
OPENAI_TEXT_MODEL = "gpt-5.6-terra"
DEFAULT_MAX_VIDEO_BYTES = 15 * 1024 * 1024
SUPPORTED_VIDEO_SUFFIXES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
ROOT = Path(__file__).resolve().parent


class CourseError(RuntimeError):
    """Base exception with messages suitable for notebook users."""


class ConfigurationError(CourseError):
    """Raised when credentials, packages, or course configuration are missing."""


class ServiceError(CourseError):
    """Raised when the RCD service cannot complete a request."""


class VideoValidationError(CourseError):
    """Raised before an invalid or excessive video is transmitted."""


def configured_base_url(explicit: str | None = None) -> str:
    """Resolve an explicit or environment-configured OpenAI-compatible endpoint."""
    return (
        explicit
        or os.getenv("COURSE_LLM_BASE_URL")
        or os.getenv("RCD_LLM_BASE_URL")
        or RCD_LLM_BASE_URL
    ).rstrip("/")


def configured_api_key(explicit: str | None = None) -> str | None:
    """Resolve the generic course key while preserving the original RCD alias."""
    return explicit or os.getenv("COURSE_LLM_API_KEY") or os.getenv("RCD_LLM_API_KEY")


def create_client(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    prompt_if_missing: bool = True,
    timeout: float = 300.0,
):
    """Create an OpenAI client for RCD or another configured compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ConfigurationError("The 'openai' package is missing. Run: python -m pip install -r requirements.txt") from exc

    key = configured_api_key(api_key)
    if not key and prompt_if_missing:
        key = getpass.getpass("Enter your RCD LLM API key (input is hidden): ").strip()
    if not key:
        raise ConfigurationError("No API key found. Set COURSE_LLM_API_KEY/RCD_LLM_API_KEY or allow the secure prompt.")
    return OpenAI(api_key=key, base_url=configured_base_url(base_url), timeout=timeout)


def create_openai_gateway_client(
    api_key: str | None = None,
    *,
    prompt_if_missing: bool = True,
    timeout: float = 300.0,
):
    """Create a client for OpenAI-hosted models reached through the RLS gateway."""
    return create_client(
        api_key,
        base_url=RCD_OPENAI_BASE_URL,
        prompt_if_missing=prompt_if_missing,
        timeout=timeout,
    )


def _model_rows(payload: Any) -> list[dict[str, Any]]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if hasattr(payload, "data") and not isinstance(payload, Mapping):
        payload = payload.data
    if isinstance(payload, Mapping):
        payload = payload.get("data", payload.get("models", []))
    rows = []
    for item in payload or []:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif not isinstance(item, Mapping):
            item = vars(item)
        rows.append(dict(item))
    return rows


def list_capable_models(
    required_features: Iterable[str] = (),
    *,
    client: Any | None = None,
    models_payload: Any | None = None,
) -> list[dict[str, Any]]:
    """Return active local models containing every requested capability.

    `models_payload` keeps the parsing logic testable without changing the live
    notebook path.
    """
    required = {feature.strip().lower() for feature in required_features}
    if models_payload is None:
        client = client or create_client()
        try:
            models_payload = client.models.list()
        except Exception as exc:
            raise ServiceError(_friendly_service_message(exc, "model catalog")) from exc

    matches = []
    for row in _model_rows(models_payload):
        lifecycle = str(row.get("lifecycle", "active")).lower()
        features = {str(value).lower() for value in row.get("features", [])}
        if lifecycle == "active" and required.issubset(features):
            matches.append(row)
    return sorted(matches, key=lambda row: str(row.get("id", "")))


def choose_model(required_features: Iterable[str], *, client: Any | None = None, models_payload: Any | None = None) -> str:
    matches = list_capable_models(required_features, client=client, models_payload=models_payload)
    if not matches:
        wanted = ", ".join(required_features) or "the requested"
        raise ConfigurationError(f"No active local model advertises all required features: {wanted}.")
    return str(matches[0]["id"])


def list_model_ids(*, client: Any) -> list[str]:
    """Return the exact model IDs advertised by one configured endpoint."""
    try:
        return sorted(str(row.get("id")) for row in _model_rows(client.models.list()) if row.get("id"))
    except Exception as exc:
        raise ServiceError(_friendly_service_message(exc, "model catalog")) from exc


def require_models(model_ids: Iterable[str], *, client: Any) -> list[str]:
    """Fail early when a model fixed by the lesson is absent from the endpoint."""
    available = set(list_model_ids(client=client))
    requested = [str(model_id) for model_id in model_ids]
    missing = [model_id for model_id in requested if model_id not in available]
    if missing:
        raise ConfigurationError(
            "These lesson models are not currently advertised by this endpoint: "
            + ", ".join(missing)
            + ". Check the live Models page before class."
        )
    return requested


def run_chat(
    messages: Sequence[Mapping[str, Any]],
    model: str,
    *,
    client: Any | None = None,
    max_tokens: int = 700,
    temperature: float = 0.2,
    tools: Sequence[Mapping[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> Any:
    """Run one live Chat Completions request against the local RCD endpoint."""
    client = client or create_client()
    request: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        request["tools"] = list(tools)
    if enable_thinking:
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    try:
        return client.chat.completions.create(**request).choices[0].message
    except Exception as exc:
        raise ServiceError(_friendly_service_message(exc, "chat completion")) from exc


def run_completion(
    prompt: str,
    model: str,
    *,
    client: Any | None = None,
    max_tokens: int = 40,
    temperature: float = 0.8,
) -> str:
    """Use the legacy-style completions endpoint to expose next-token behavior."""
    client = client or create_client()
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].text
    except Exception as exc:
        raise ServiceError(_friendly_service_message(exc, "text completion")) from exc


def run_openai_response(
    prompt: str,
    *,
    client: Any | None = None,
    model: str = OPENAI_TEXT_MODEL,
    max_output_tokens: int = 300,
) -> str:
    """Send a text-only Responses API request through the RLS OpenAI gateway."""
    client = client or create_openai_gateway_client()
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        raise ServiceError(_friendly_service_message(exc, "OpenAI text response")) from exc
    text = str(getattr(response, "output_text", "") or "").strip()
    if not text:
        raise ServiceError("The OpenAI gateway returned no text in response.output_text.")
    return text


def _message_mapping(message: Any) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return vars(message)


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        pieces = []
        for item in value:
            if isinstance(item, Mapping):
                pieces.append(str(item.get("text", item.get("content", ""))))
            else:
                pieces.append(str(getattr(item, "text", item)))
        return "".join(pieces)
    return str(value)


def message_parts(message: Any) -> dict[str, str]:
    """Return thinking and final-answer text without mixing the two channels."""
    row = _message_mapping(message)
    reasoning = row.get("reasoning_content", row.get("reasoning", ""))
    return {
        "reasoning": _content_text(reasoning),
        "content": _content_text(row.get("content", "")),
    }


def message_content(message: Any) -> str:
    """Return only the final answer; use ``message_parts`` to inspect thinking."""
    return message_parts(message)["content"]


def redact_request(value: Any) -> Any:
    """Return a JSON-safe copy with likely secrets removed."""
    secret_names = {"authorization", "api_key", "apikey", "token", "password"}
    if isinstance(value, Mapping):
        return {key: ("<redacted>" if str(key).lower() in secret_names else redact_request(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_request(item) for item in value]
    return value


def validate_video_path(video_path: str | Path, *, max_bytes: int = DEFAULT_MAX_VIDEO_BYTES) -> tuple[Path, str]:
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise VideoValidationError(f"Video not found: {path}")
    mime = SUPPORTED_VIDEO_SUFFIXES.get(path.suffix.lower())
    if not mime:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise VideoValidationError(f"Unsupported video format '{path.suffix}'. Use one of: {supported}.")
    size = path.stat().st_size
    if size == 0:
        raise VideoValidationError("The video file is empty.")
    if size > max_bytes:
        raise VideoValidationError(
            f"Video is {size / 1024 / 1024:.1f} MB; the classroom limit is {max_bytes / 1024 / 1024:.1f} MB. "
            "Use the instructor-provided short clip."
        )
    return path, mime


def video_data_url(video_path: str | Path, *, max_bytes: int = DEFAULT_MAX_VIDEO_BYTES) -> str:
    path, mime = validate_video_path(video_path, max_bytes=max_bytes)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


VIDEO_ANALYSIS_PROMPT = """Analyze this short video. Return JSON only with this structure:
{
  "summary": "concise description",
  "events": [
    {"start_seconds": 0, "end_seconds": 5, "description": "observable event", "evidence": "visible evidence"}
  ],
  "uncertainties": ["details that cannot be determined reliably"]
}
Keep events chronological. Use estimated timestamps, distinguish observation from inference, and do not invent unseen details.
"""


def analyze_video(
    video_path: str | Path,
    prompt: str,
    model: str,
    *,
    client: Any | None = None,
    max_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
    max_tokens: int = 1200,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """Analyze a short video, optionally using Qwen's separate thinking channel.

    ``max_tokens`` limits all newly generated output. A thinking request therefore
    needs a substantially larger value than a direct, non-thinking JSON request.
    """
    path, _ = validate_video_path(video_path, max_bytes=max_bytes)
    client = client or create_client()

    messages = [{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_data_url(path, max_bytes=max_bytes)}},
            {"type": "text", "text": prompt},
        ],
    }]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            # Never depend on the server default: the two notebook experiments
            # deliberately choose whether Qwen3-Omni uses its thinking channel.
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
    except Exception as exc:
        raise ServiceError(_friendly_service_message(exc, "video analysis")) from exc

    choice = response.choices[0]
    parts = message_parts(choice.message)
    if not parts["content"].strip():
        finish_reason = getattr(choice, "finish_reason", None)
        if parts["reasoning"].strip():
            advice = (
                "Increase max_tokens so there is room for both reasoning and the final JSON answer."
                if enable_thinking
                else
                "Restart the notebook kernel and rerun the setup/import cells so the updated helper is loaded."
            )
            raise ServiceError(
                "The video model returned reasoning but no final JSON answer. "
                + advice + " "
                f"The server finish_reason was {finish_reason!r}."
            )
        raise ServiceError(
            "The video model returned no text in message.content. "
            "Check that the selected model supports video input, then inspect the "
            f"complete response. The server finish_reason was {finish_reason!r}."
        )
    return validate_video_analysis(parse_json_object(parts["content"]))


def parse_json_object(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        raise ServiceError("The model returned no text to parse as JSON.")
    if not isinstance(value, str):
        raise ServiceError(
            "Expected the model response to be text or a JSON object, "
            f"but received {type(value).__name__}."
        )
    text = value.strip()
    if not text:
        raise ServiceError("The model returned an empty string instead of a JSON object.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ServiceError(f"The model did not return valid JSON: {exc.msg} at character {exc.pos}.") from exc
    if not isinstance(parsed, dict):
        raise ServiceError("The model returned JSON, but the top-level value is not an object.")
    return parsed


def validate_video_analysis(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if not isinstance(result.get("summary"), str) or not result["summary"].strip():
        raise ServiceError("Video result requires a non-empty string 'summary'.")
    if not isinstance(result.get("events"), list):
        raise ServiceError("Video result requires an 'events' list.")
    if not isinstance(result.get("uncertainties"), list) or not all(isinstance(x, str) for x in result["uncertainties"]):
        raise ServiceError("Video result requires an 'uncertainties' list of strings.")
    previous_start = -1.0
    for index, event in enumerate(result["events"]):
        if not isinstance(event, Mapping):
            raise ServiceError(f"Event {index} must be an object.")
        required = {"start_seconds", "end_seconds", "description", "evidence"}
        if not required.issubset(event):
            raise ServiceError(f"Event {index} is missing: {', '.join(sorted(required - set(event)))}.")
        start, end = event["start_seconds"], event["end_seconds"]
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start < 0 or end < start:
            raise ServiceError(f"Event {index} has an invalid time range.")
        if start < previous_start:
            raise ServiceError("Events must be chronological.")
        if not isinstance(event["description"], str) or not isinstance(event["evidence"], str):
            raise ServiceError(f"Event {index} description and evidence must be strings.")
        previous_start = float(start)
    return result


def compare_events(predicted: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]], tolerance_seconds: float = 2.0) -> list[dict[str, Any]]:
    """Create a transparent, human-reviewable comparison; this is not an accuracy score."""
    rows = []
    for index, expected in enumerate(reference):
        candidate = predicted[index] if index < len(predicted) else None
        within = False
        if candidate:
            within = abs(float(candidate["start_seconds"]) - float(expected["start_seconds"])) <= tolerance_seconds
        rows.append({
            "reference_event": expected.get("description"),
            "model_event": candidate.get("description") if candidate else "<missing>",
            "start_within_tolerance": within,
            "requires_human_review": True,
        })
    if len(predicted) > len(reference):
        for candidate in predicted[len(reference):]:
            rows.append({"reference_event": "<none>", "model_event": candidate.get("description"), "start_within_tolerance": False, "requires_human_review": True})
    return rows


def temporal_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Intersection-over-union for two temporal intervals."""
    start = max(float(first["start_seconds"]), float(second["start_seconds"]))
    end = min(float(first["end_seconds"]), float(second["end_seconds"]))
    intersection = max(0.0, end - start)
    union = max(float(first["end_seconds"]), float(second["end_seconds"])) - min(
        float(first["start_seconds"]), float(second["start_seconds"])
    )
    return intersection / union if union > 0 else 0.0


def evaluate_video_events(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Greedily match events and preserve rows for required human review."""
    remaining = set(range(len(reference)))
    rows = []
    for candidate in predicted:
        scored = [(temporal_iou(candidate, reference[index]), index) for index in remaining]
        overlap, match = max(scored, default=(0.0, None))
        accepted = match is not None and overlap >= iou_threshold
        if accepted:
            remaining.remove(match)
        rows.append({
            "model_event": candidate.get("description", ""),
            "reference_event": reference[match].get("description", "") if accepted else "<none>",
            "temporal_iou": overlap,
            "status": "matched" if accepted else "extra_or_shifted",
            "requires_human_review": True,
        })
    for index in sorted(remaining):
        rows.append({
            "model_event": "<missing>",
            "reference_event": reference[index].get("description", ""),
            "temporal_iou": 0.0,
            "status": "missing",
            "requires_human_review": True,
        })
    matched = sum(row["status"] == "matched" for row in rows)
    return {
        "matched": matched,
        "missing": sum(row["status"] == "missing" for row in rows),
        "extra_or_shifted": sum(row["status"] == "extra_or_shifted" for row in rows),
        "event_recall": matched / max(len(reference), 1),
        "rows": rows,
    }


def execute_authorized_tool(
    name: str,
    arguments: Mapping[str, Any],
    *,
    registry: Mapping[str, Callable[..., str]],
    allowed_tools: Iterable[str],
    confirmed: bool = False,
    confirmation_required: Iterable[str] = (),
    max_output_chars: int = 2000,
) -> dict[str, Any]:
    """Validate authority and confirmation before executing a bounded tool."""
    allowed = set(allowed_tools)
    confirmation_set = set(confirmation_required)
    if name not in allowed or name not in registry:
        raise PermissionError(f"Tool is not authorized: {name}")
    if name in confirmation_set and not confirmed:
        raise PermissionError(f"Tool requires confirmation immediately before execution: {name}")
    try:
        result = registry[name](**dict(arguments))
        text = str(result)
        return {"tool": name, "ok": True, "output": text[:max_output_chars], "truncated": len(text) > max_output_chars}
    except Exception as exc:
        return {"tool": name, "ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}


class TraceLog:
    """Minimal append-only classroom trace with idempotency-key detection."""
    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.completed_keys: set[str] = set()

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {"sequence": len(self.events), "event": event, **redact_request(fields)}
        self.events.append(row)
        return row

    def begin_action(self, idempotency_key: str) -> bool:
        if idempotency_key in self.completed_keys:
            self.record("duplicate_suppressed", idempotency_key=idempotency_key)
            return False
        self.record("action_started", idempotency_key=idempotency_key)
        return True

    def complete_action(self, idempotency_key: str) -> None:
        self.completed_keys.add(idempotency_key)
        self.record("action_completed", idempotency_key=idempotency_key)


_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARYOPS: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculator(expression: str) -> str:
    """Evaluate arithmetic only—never Python names, calls, attributes, or imports."""
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic is allowed.")
    if len(expression) > 200:
        raise ValueError("Expression is too long.")
    return str(evaluate(ast.parse(expression, mode="eval")))


def earthquake_lookup(event_id: str, path: str | Path | None = None) -> str:
    """Return one recorded USGS earthquake from the packaged January 2025 snapshot."""
    source = Path(path) if path else ROOT / "data" / "usgs_earthquakes_2025_01.jsonl"
    event_id = str(event_id).strip()
    if not event_id:
        raise ValueError("event_id must be a nonempty USGS event identifier")
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event_id"] == event_id:
            return json.dumps(row, ensure_ascii=False, sort_keys=True)
    raise KeyError(f"USGS event not found in packaged snapshot: {event_id}")


def _friendly_service_message(exc: Exception, operation: str) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "authentication" in name or "401" in text:
        return f"RCD authentication failed during {operation}. Check that the API key and allocation are active."
    if "timeout" in name or "timed out" in text:
        return f"RCD timed out during {operation}. Retry once with the short classroom input."
    if "connection" in name or "connect" in text:
        return f"Could not reach RCD during {operation}. Confirm campus networking or CUVPN access."
    if "429" in text or "rate" in text:
        return f"RCD is temporarily rate-limiting {operation}. Pause briefly, then retry once."
    if "policy_denied" in text and any(word in text for word in ("image", "file", "stored-state", "hosted tool")):
        return (
            f"RLS policy denied {operation}: image/file and related stored or hosted inputs "
            "are blocked on the OpenAI gateway. Use the text-only gateway example or a "
            "supported local RLS multimodal model."
        )
    if "permission" in name or "403" in text or "policy_denied" in text:
        return (
            f"RLS denied {operation}. Check the OpenAI Credits page for the recorded "
            "reason, including allocation, remaining credits, model availability, and policy."
        )
    return f"RCD could not complete {operation}: {type(exc).__name__}. Ask the instructor if it persists."
