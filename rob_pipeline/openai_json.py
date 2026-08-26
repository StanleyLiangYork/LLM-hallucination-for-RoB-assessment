"""OpenAI JSON calls with structured-output and legacy-model fallbacks."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, BadRequestError, OpenAI, RateLimitError


@dataclass(frozen=True)
class JsonCallResult:
    data: dict[str, Any]
    raw_text: str
    response_id: str | None
    finish_reason: str | None
    output_mode: str


def _is_reasoning_model(model: str) -> bool:
    name = model.casefold().strip()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def _extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON response must be an object")
                return value
    raise ValueError("Model response contained an incomplete JSON object")


class OpenAIJsonClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        max_retries: int = 5,
        timeout: float = 180.0,
    ) -> None:
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> JsonCallResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        modes: list[tuple[str, dict[str, Any] | None]] = [
            (
                "json_schema",
                {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            ),
            ("json_object", {"type": "json_object"}),
            ("prompt_only", None),
        ]
        last_error: Exception | None = None
        for mode, response_format in modes:
            try:
                return self._call(messages, response_format, mode)
            except BadRequestError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise last_error

    def _call(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        mode: str,
    ) -> JsonCallResult:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if not _is_reasoning_model(self.model):
            kwargs["temperature"] = 0

        removed_optional = False
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                raw_text = choice.message.content or ""
                return JsonCallResult(
                    data=_extract_json(raw_text),
                    raw_text=raw_text,
                    response_id=getattr(response, "id", None),
                    finish_reason=getattr(choice, "finish_reason", None),
                    output_mode=mode,
                )
            except BadRequestError as exc:
                message = str(exc).casefold()
                if not removed_optional and (
                    "reasoning_effort" in message or "temperature" in message
                ):
                    kwargs.pop("reasoning_effort", None)
                    kwargs.pop("temperature", None)
                    removed_optional = True
                    continue
                raise
            except (RateLimitError, APIConnectionError, APIError):
                if attempt >= self.max_retries:
                    raise
                delay = min(60.0, (2**attempt) + random.random())
                time.sleep(delay)
